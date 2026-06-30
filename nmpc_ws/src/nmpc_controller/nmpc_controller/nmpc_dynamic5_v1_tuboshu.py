import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64
from nmpc_interfaces.msg import CpsTrajectory
from visualization_msgs.msg import Marker, MarkerArray
import casadi as ca
import numpy as np
import math
from scipy.interpolate import interp1d
import time


class NMPCLeadLagNode(Node):
    def __init__(self):
        super().__init__('nmpc_lead_lag_node')

        self.dt = 0.05
        self.control_rate = 1.0 / self.dt
        self.last_path_signature = None

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        #self.path_sub = self.create_subscription(Path, '/global_path', self.path_callback, 10)
        self.cps_traj_sub = self.create_subscription(CpsTrajectory, '/cps_trajectory', self.cps_trajectory_callback, 10)
        self.cps_ref_path_pub = self.create_publisher(Path, '/cps_reference_path', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc_path', 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.x_act = 0.0
        self.y_act = 0.0
        self.theta_act = 0.0

        self.earth_R = 6378137.0      #地球半径

        self.cps_origin_set = False    #是否已经记录CPS轨迹原点
        self.cps_lat0 = 0.0            #设定局部坐标原点
        self.cps_lon0 = 0.0
        self.cps_alt0 = 0.0

        self.cps_odom_x0 = 0.0         #CPS原点平移到当前/odom位置附近
        self.cps_odom_y0 = 0.0
        self.cps_odom_yaw0 = 0.0       #方向旋转对齐

        self.align_cps_to_current_odom = True      
        
        self.v_act = 0.0
        self.vy_act = 0.0
        self.omega_act = 0.0

        self.m_vehicle = 190.0             
        self.Iz_vehicle = 35.0         
        self.R_wheel = 0.315            
        self.B_track = 0.910            
        self.L_wheelbase = 0.795      
        self.h_cg = 0.250               
        self.mu_nominal = 0.6           
        self.g_const = 9.81

        self.kv_nominal = 1.0
        self.kr_nominal = 1.0
        self.tau_v = 0.045
        self.tau_r = 0.055

        self.slope_gain = 0.0
        self.cross_slope_gain = 0.0
        self.tau_y = 3.0
        self.ky_vr = 0.0
        self.k_vxr = 0.0

        self.ll_v_eps = 0.15
        self.ll_time_err_max = 0.8
        self.ll_K_time_v = 1.0
        self.ll_dv_max = 0.50

        # 检查点软约束参数
        self.use_checkpoint_soft_constraint = True

        # 检查点权重。先不要太大，避免一开始就让速度饱和
        self.cp_weight = 300.0

        # 检查点时间，单位 s。
        # 这里先用 None，等收到轨迹后自动生成。
        self.checkpoint_times = None

        # 检查点实际通过判定半径，用于后续统计，不直接参与 NMPC
        self.cp_arrival_radius = 0.05

        # 记录检查点实际到达时间，后面做实验统计
        self.cp_arrival_logged = {}

        self.use_slope_feedforward = False
        self.ff_slope_gain = 0.0

        self.C_y = self.m_vehicle / self.tau_y          
        self.C_yr = 0.0       

        self.c_rr = 0.0                  
        self.c_d = 0.0                
        self.C_r = 0.0              
        self.C_rv = 0.0                  
        
        self.sat_eps = 1e-6
        self.smooth_sign_eps = 0.05
        self.smooth_abs_eps = 1e-6

        self.max_slope_rad = 0.45      
        self.max_beta_rad = 0.45         
        self.mu_min = 0.1           
        self.mu_max = 1.5
        self.alpha_filter_window = 5      
        self.beta_filter_window = 5      
        self.use_path_roll_as_beta = True
        self.beta_default = 0.0           

        self.path_ready = False
        self.t_track = None         
        self.t_final = 0.0         
        self.start_time = None    
        self.t_ref_offset = 0.0     

        self.u_prev = np.array([0.0, 0.0])

        self.is_initialized = False

        self.get_logger().info("初始化 CasADi NMPC 求解器：版本4 6状态物理灰箱动力学...")
        self.init_casadi_solver()
        self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)
        self.get_logger().info("NMPC Lead-Lag 动力学节点已启动！等待接收全局路径...")

        self.actual_path_pub = self.create_publisher(Path, '/actual_path', 10)

        self.et_pub = self.create_publisher(Float64, '/nmpc_debug/e_t', 10)
        self.en_pub = self.create_publisher(Float64, '/nmpc_debug/e_n', 10)
        self.eth_pub = self.create_publisher(Float64, '/nmpc_debug/e_theta', 10)
        self.tref_pub = self.create_publisher(Float64, '/nmpc_debug/t_ref', 10)
        self.goal_dist_pub = self.create_publisher(Float64, '/nmpc_debug/goal_dist', 10)
        self.solve_ms_pub = self.create_publisher(Float64, '/nmpc_debug/solve_ms', 10)
        self.vcmd_pub = self.create_publisher(Float64, '/nmpc_debug/v_cmd', 10)
        self.wcmd_pub = self.create_publisher(Float64, '/nmpc_debug/omega_cmd', 10)
        self.v_sat_ratio_pub = self.create_publisher(Float64, '/nmpc_debug/v_sat_ratio', 10)
        self.w_sat_ratio_pub = self.create_publisher(Float64, '/nmpc_debug/omega_sat_ratio', 10)
        self.alpha_pub = self.create_publisher(Float64, '/nmpc_debug/alpha', 10)
        self.beta_pub = self.create_publisher(Float64, '/nmpc_debug/beta', 10)
        self.mu_pub = self.create_publisher(Float64, '/nmpc_debug/mu', 10)
        self.vy_pub = self.create_publisher(Float64, '/nmpc_debug/vy', 10)
        self.checkpoint_marker_pub = self.create_publisher(MarkerArray, '/checkpoint_markers', 10)

        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = "odom"

        self.total_ctrl_count = 0
        self.v_sat_count = 0
        self.w_sat_count = 0
        self.arrival_time = None

        self.odom_ready = False

    def publish_scalar(self, pub, value: float):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    def publish_cps_reference_path_for_rviz(self, x_pts, y_pts, theta_pts, t_pts):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"

        for i in range(len(x_pts)):
            pose = PoseStamped()
            pose.header = path_msg.header

            pose.pose.position.x = float(x_pts[i])
            pose.pose.position.y = float(y_pts[i])
            pose.pose.position.z = 0.0

            yaw = float(theta_pts[i])
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            path_msg.poses.append(pose)

        self.cps_ref_path_pub.publish(path_msg)

    def publish_checkpoint_markers_for_rviz(self):
        if self.checkpoint_times is None or self.traj_interp is None:
            return

        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        for idx, t_cp in enumerate(self.checkpoint_times):
            x_cp = float(self.traj_interp['x_of_t'](t_cp))
            y_cp = float(self.traj_interp['y_of_t'](t_cp))

            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now_msg
            marker.ns = "checkpoints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = x_cp
            marker.pose.position.y = y_cp
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = "odom"
            text.header.stamp = now_msg
            text.ns = "checkpoint_labels"
            text.id = 1000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = x_cp
            text.pose.position.y = y_cp
            text.pose.position.z = 0.45
            text.pose.orientation.w = 1.0

            text.scale.z = 0.25
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = f"{t_cp:.1f}s"

            marker_array.markers.append(text)

        self.checkpoint_marker_pub.publish(marker_array)

    def init_casadi_solver(self):
        self.Np = 20
        self.nx = 6       
        self.nu = 2      
        self.np_terrain = 3  

        self.v_min, self.v_max = -0.5, 2.0
        self.omega_min, self.omega_max = -math.pi / 2, math.pi / 2

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        terrain = ca.SX.sym('terrain', self.np_terrain)

        theta = x[2]
        vx = x[3]
        vy = x[4]
        r = x[5]

        v_cmd = u[0]
        omega_cmd = u[1]

        alpha = terrain[0]  
        beta = terrain[1]  
        mu = terrain[2]   

        m_vehicle = self.m_vehicle
        Iz_vehicle = self.Iz_vehicle
        B_track = self.B_track
        kv = self.kv_nominal
        kr = self.kr_nominal
        tau_v = self.tau_v
        tau_r = self.tau_r
        g_const = self.g_const
        slope_gain = self.slope_gain
        cross_slope_gain = self.cross_slope_gain
        c_rr = self.c_rr
        c_d = self.c_d
        C_y = self.C_y
        C_yr = self.C_yr
        C_r = self.C_r
        C_rv = self.C_rv
        sat_eps = self.sat_eps
        smooth_sign_eps = self.smooth_sign_eps
        smooth_abs_eps = self.smooth_abs_eps

        X_dot = vx * ca.cos(theta) - vy * ca.sin(theta)
        Y_dot = vx * ca.sin(theta) + vy * ca.cos(theta)
        theta_dot = r

        mu_eff = ca.fmax(self.mu_min, ca.fmin(self.mu_max, mu))
        gamma = ca.sqrt(alpha**2 + beta**2 + 1e-8)
        normal_factor = ca.cos(gamma)

        Fx_raw = m_vehicle / tau_v * (kv * v_cmd - vx)
        Fx_max = mu_eff * m_vehicle * g_const * normal_factor
        Fx = Fx_max * ca.tanh(Fx_raw / (Fx_max + sat_eps))

        smooth_sign_vx = ca.tanh(vx / smooth_sign_eps)
        smooth_abs_vx = ca.sqrt(vx**2 + smooth_abs_eps)
        F_roll = c_rr * m_vehicle * g_const * normal_factor * smooth_sign_vx
        F_drag = c_d * vx * smooth_abs_vx

        Fy = -C_y * vy - C_yr * r

        Mz_raw = Iz_vehicle / tau_r * (kr * omega_cmd - r)
        Mz_max = mu_eff * m_vehicle * g_const * normal_factor * B_track / 2.0
        Mz = Mz_max * ca.tanh(Mz_raw / (Mz_max + sat_eps))

        M_res = C_r * r + C_rv * vy

        k_vxr = self.k_vxr
        ky_vr = self.ky_vr
        vx_dot = vy * r + (Fx - F_roll - F_drag) / m_vehicle - slope_gain * g_const * ca.sin(alpha)
        vy_dot = (-k_vxr * vx * r + ky_vr * vx * r + Fy / m_vehicle - cross_slope_gain * g_const * ca.sin(beta))
        r_dot = (Mz - M_res) / Iz_vehicle

        f_continuous = ca.Function(
            'f_continuous',
            [x, u, terrain],
            [ca.vertcat(X_dot, Y_dot, theta_dot, vx_dot, vy_dot, r_dot)]
        )

        k1 = f_continuous(x, u, terrain)
        k2 = f_continuous(x + self.dt / 2 * k1, u, terrain)
        k3 = f_continuous(x + self.dt / 2 * k2, u, terrain)
        k4 = f_continuous(x + self.dt * k3, u, terrain)
        x_next = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F_discrete = ca.Function('F_discrete', [x, u, terrain], [x_next])

        X = ca.SX.sym('X', self.nx, self.Np + 1)
        U = ca.SX.sym('U', self.nu, self.Np)
        X_ref = ca.SX.sym('X_ref', self.nx, self.Np + 1)
        U_ref = ca.SX.sym('U_ref', self.nu, self.Np)
        Terrain_ref = ca.SX.sym('Terrain_ref', self.np_terrain, self.Np)

        # 新增：检查点参考
        # 第0行：检查点x
        # 第1行：检查点y
        # 第2行：检查点权重，0表示该预测步没有检查点
        CP_ref = ca.SX.sym('CP_ref', 3, self.Np + 1)

        x0 = ca.SX.sym('x0', self.nx)
        u_prev = ca.SX.sym('u_prev', self.nu)

        J = 0
        g = [X[:, 0] - x0]

        Qdu = ca.DM(np.diag([2.0, 1.0]))

        for k in range(self.Np):
            dx = X[0, k] - X_ref[0, k]
            dy = X[1, k] - X_ref[1, k]
            th_ref = X_ref[2, k]

            e_t = dx * ca.cos(th_ref) + dy * ca.sin(th_ref)
            e_n = -dx * ca.sin(th_ref) + dy * ca.cos(th_ref)
            e_theta = ca.atan2(ca.sin(X[2, k] - th_ref), ca.cos(X[2, k] - th_ref))

            vx_err = X[3, k] - X_ref[3, k]
            vy_err = X[4, k] - X_ref[4, k]
            r_err = X[5, k] - X_ref[5, k]

            v_err = U[0, k] - U_ref[0, k]
            w_err = U[1, k] - U_ref[1, k]

            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])

            J += 40.0 * e_t**2 + 80.0 * e_n**2 + 8.0 * e_theta**2
            J += 3.0 * vx_err**2 + 12.0 * vy_err**2 + 2.0 * r_err**2
            J += 1.0 * v_err**2 + 0.5 * w_err**2
            J += ca.mtimes([du.T, Qdu, du])

            # 检查点软约束
            cp_x = CP_ref[0, k]
            cp_y = CP_ref[1, k]
            cp_w = CP_ref[2, k]
            cp_err2 = (X[0, k] - cp_x)**2 + (X[1, k] - cp_y)**2
            J += cp_w * cp_err2

            g.append(X[:, k + 1] - F_discrete(X[:, k], U[:, k], Terrain_ref[:, k]))

        dxN = X[0, self.Np] - X_ref[0, self.Np]
        dyN = X[1, self.Np] - X_ref[1, self.Np]
        th_refN = X_ref[2, self.Np]

        e_tN = dxN * ca.cos(th_refN) + dyN * ca.sin(th_refN)
        e_nN = -dxN * ca.sin(th_refN) + dyN * ca.cos(th_refN)
        e_thetaN = ca.atan2(ca.sin(X[2, self.Np] - th_refN), ca.cos(X[2, self.Np] - th_refN))
        vx_errN = X[3, self.Np] - X_ref[3, self.Np]
        vy_errN = X[4, self.Np] - X_ref[4, self.Np]
        r_errN = X[5, self.Np] - X_ref[5, self.Np]

        J += 120.0 * e_tN**2 + 150.0 * e_nN**2 + 12.0 * e_thetaN**2
        J += 6.0 * vx_errN**2 + 25.0 * vy_errN**2 + 4.0 * r_errN**2

        # 终端步检查点软约束
        cp_xN = CP_ref[0, self.Np]
        cp_yN = CP_ref[1, self.Np]
        cp_wN = CP_ref[2, self.Np]

        cp_err2N = (X[0, self.Np] - cp_xN)**2 + (X[1, self.Np] - cp_yN)**2
        J += cp_wN * cp_err2N

        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = ca.vertcat(
            ca.reshape(X_ref, -1, 1),
            ca.reshape(U_ref, -1, 1),
            ca.reshape(Terrain_ref, -1, 1),
            ca.reshape(CP_ref, -1, 1),
            x0,
            u_prev
        )

        nlp = {'x': opt_variables, 'p': opt_params, 'f': J, 'g': ca.vertcat(*g)}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbg = np.zeros(self.nx * (self.Np + 1))
        self.ubg = np.zeros(self.nx * (self.Np + 1))

        lbx = np.full(self.nx * (self.Np + 1), -np.inf)
        ubx = np.full(self.nx * (self.Np + 1), np.inf)

        self.vy_min, self.vy_max = -1.0, 1.0
        for k in range(self.Np + 1):
            idx_vx = k * self.nx + 3
            idx_vy = k * self.nx + 4
            idx_r = k * self.nx + 5
            lbx[idx_vx] = self.v_min
            ubx[idx_vx] = self.v_max
            lbx[idx_vy] = self.vy_min
            ubx[idx_vy] = self.vy_max
            lbx[idx_r] = self.omega_min
            ubx[idx_r] = self.omega_max

        lbu = np.tile([self.v_min, self.omega_min], self.Np)
        ubu = np.tile([self.v_max, self.omega_max], self.Np)

        self.lb_opt = np.concatenate([lbx, lbu])
        self.ub_opt = np.concatenate([ubx, ubu])

    def odom_callback(self, msg):
        self.x_act = msg.pose.pose.position.x
        self.y_act = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta_act = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.v_act = msg.twist.twist.linear.x
        self.vy_act = msg.twist.twist.linear.y
        self.omega_act = msg.twist.twist.angular.z
        self.odom_ready = True

    def gps_rad_to_local_xy(self, lat_rad, lon_rad):
        dx_east = (lon_rad - self.cps_lon0) * self.earth_R * math.cos(self.cps_lat0)
        dy_north = (lat_rad - self.cps_lat0) * self.earth_R
        return dx_east, dy_north
    
    def local_xy_to_odom_xy(self, x_local, y_local):
        c = math.cos(self.cps_odom_yaw0)
        s = math.sin(self.cps_odom_yaw0)

        x_odom = self.cps_odom_x0 + c * x_local - s * y_local
        y_odom = self.cps_odom_y0 + s * x_local + c * y_local

        return x_odom, y_odom

    #  def path_callback(self, msg):
    #     if len(msg.poses) < 2:
    #         return

    #     self.get_logger().info("收到带时间戳参考轨迹，直接建立时间插值...")

    #     x_pts, y_pts, z_pts, theta_pts, roll_pts, t_pts = [], [], [], [], [], []
    #     t0_sec = None
    #     for pose_stamped in msg.poses:
    #         x = pose_stamped.pose.position.x
    #         y = pose_stamped.pose.position.y
    #         z = pose_stamped.pose.position.z

    #         q = pose_stamped.pose.orientation
    #         theta = math.atan2(
    #             2.0 * (q.w * q.z + q.x * q.y),
    #             1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    #         )
    #         roll = math.atan2(
    #             2.0 * (q.w * q.x + q.y * q.z),
    #             1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    #         )

    #         sec = pose_stamped.header.stamp.sec + pose_stamped.header.stamp.nanosec * 1e-9
    #         if t0_sec is None:
    #             t0_sec = sec
    #         t_rel = sec - t0_sec

    #         x_pts.append(x)
    #         y_pts.append(y)
    #         z_pts.append(z)
    #         theta_pts.append(theta)
    #         roll_pts.append(roll)
    #         t_pts.append(t_rel)

    #     x_pts = np.array(x_pts)
    #     y_pts = np.array(y_pts)
    #     z_pts = np.array(z_pts)
    #     theta_pts = np.unwrap(np.array(theta_pts))
    #     roll_pts = np.unwrap(np.array(roll_pts))
    #     t_pts = np.array(t_pts)

    #     keep = np.hstack(([True], np.diff(t_pts) > 1e-6))
    #     x_pts = x_pts[keep]
    #     y_pts = y_pts[keep]
    #     z_pts = z_pts[keep]
    #     theta_pts = theta_pts[keep]
    #     roll_pts = roll_pts[keep]
    #     t_pts = t_pts[keep]

    #     if len(x_pts) < 2:
    #         self.get_logger().warn("路径有效点不足，忽略本次路径")
    #         return

    #     dt = np.diff(t_pts)
    #     dx = np.diff(x_pts)
    #     dy = np.diff(y_pts)
    #     dz = np.diff(z_pts)
    #     dtheta = np.diff(theta_pts)

    #     ds = np.sqrt(dx**2 + dy**2)
    #     alpha_pts = np.zeros_like(t_pts)
    #     alpha_pts[1:] = np.arctan2(dz, np.maximum(ds, 1e-6))
    #     alpha_pts[0] = alpha_pts[1]

    #     alpha_pts = np.clip(alpha_pts, -self.max_slope_rad, self.max_slope_rad)

    #     if self.alpha_filter_window > 1 and len(alpha_pts) >= self.alpha_filter_window:
    #         kernel = np.ones(self.alpha_filter_window) / self.alpha_filter_window
    #         alpha_smooth = np.convolve(alpha_pts, kernel, mode='same')
    #         half = self.alpha_filter_window // 2
    #         alpha_smooth[:half] = alpha_pts[:half]
    #         alpha_smooth[-half:] = alpha_pts[-half:]
    #         alpha_pts = alpha_smooth

    #     if self.use_path_roll_as_beta:
    #         beta_pts = np.array(roll_pts, dtype=float)
    #     else:
    #         beta_pts = np.full_like(t_pts, self.beta_default, dtype=float)

    #     beta_pts = np.clip(beta_pts, -self.max_beta_rad, self.max_beta_rad)

    #     if self.beta_filter_window > 1 and len(beta_pts) >= self.beta_filter_window:
    #         kernel = np.ones(self.beta_filter_window) / self.beta_filter_window
    #         beta_smooth = np.convolve(beta_pts, kernel, mode='same')
    #         half = self.beta_filter_window // 2
    #         beta_smooth[:half] = beta_pts[:half]
    #         beta_smooth[-half:] = beta_pts[-half:]
    #         beta_pts = beta_smooth

    #     mu_pts = np.full_like(t_pts, self.mu_nominal, dtype=float)
    #     mu_pts = np.clip(mu_pts, self.mu_min, self.mu_max)

    #     v_pts = np.zeros_like(t_pts)
    #     omega_pts = np.zeros_like(t_pts)

    #     v_pts[1:] = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
    #     v_pts[0] = v_pts[1]

    #     omega_pts[1:] = dtheta / np.maximum(dt, 1e-6)
    #     omega_pts[0] = omega_pts[1]

    #     self.t_track = t_pts
    #     self.t_final = float(t_pts[-1])

    #     self.traj_interp = {
    #         'x_of_t': interp1d(t_pts, x_pts, bounds_error=False, fill_value=(x_pts[0], x_pts[-1])),
    #         'y_of_t': interp1d(t_pts, y_pts, bounds_error=False, fill_value=(y_pts[0], y_pts[-1])),
    #         'theta_of_t': interp1d(t_pts, theta_pts, bounds_error=False, fill_value=(theta_pts[0], theta_pts[-1])),
    #         'v_of_t': interp1d(t_pts, v_pts, bounds_error=False, fill_value=(v_pts[0], v_pts[-1])),
    #         'omega_of_t': interp1d(t_pts, omega_pts, bounds_error=False, fill_value=(omega_pts[0], omega_pts[-1])),
    #         'alpha_of_t': interp1d(t_pts, alpha_pts, bounds_error=False, fill_value=(alpha_pts[0], alpha_pts[-1])),
    #         'beta_of_t': interp1d(t_pts, beta_pts, bounds_error=False, fill_value=(beta_pts[0], beta_pts[-1])),
    #         'mu_of_t': interp1d(t_pts, mu_pts, bounds_error=False, fill_value=(mu_pts[0], mu_pts[-1])),
    #     }

    #     self.get_logger().info(
    #         f"地形预瞄已建立: alpha_min={np.min(alpha_pts):.4f} rad, "
    #         f"alpha_max={np.max(alpha_pts):.4f} rad, "
    #         f"beta_min={np.min(beta_pts):.4f} rad, "
    #         f"beta_max={np.max(beta_pts):.4f} rad, "
    #         f"mu={self.mu_nominal:.2f}"
    #     )

    #     self.path_ready = True
    #     self.is_initialized = False

    #     self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)
    #     self.actual_path_msg = Path()
    #     self.actual_path_msg.header.frame_id = "odom"
    #     self.arrival_time = None
    #     self.total_ctrl_count = 0
    #     self.v_sat_count = 0
    #     self.w_sat_count = 0

    def cps_trajectory_callback(self, msg):
        if len(msg.points) < 2:
            return

        if not self.odom_ready:
            self.get_logger().warn("收到 CPS 轨迹，但 /odom 尚未就绪，暂不处理")
            return

        self.get_logger().info("收到 CPS 经纬度时空轨迹，转换到 odom 坐标并建立时间插值...")

        lat_pts = np.array([p.latitude_rad for p in msg.points], dtype=float)
        lon_pts = np.array([p.longitude_rad for p in msg.points], dtype=float)
        z_raw = np.array([p.altitude_m for p in msg.points], dtype=float)
        v_kmh = np.array([p.speed_kmh for p in msg.points], dtype=float)
        t_pts = np.array([p.time_s for p in msg.points], dtype=float)

        # 1. 按时间去重
        keep = np.hstack(([True], np.diff(t_pts) > 1e-6))
        lat_pts = lat_pts[keep]
        lon_pts = lon_pts[keep]
        z_raw = z_raw[keep]
        v_kmh = v_kmh[keep]
        t_pts = t_pts[keep]

        if len(t_pts) < 2:
            self.get_logger().warn("CPS 轨迹有效点不足，忽略")
            return

        # 2. 以第一点为 GPS 局部坐标原点
        self.cps_lat0 = float(lat_pts[0])
        self.cps_lon0 = float(lon_pts[0])
        self.cps_alt0 = float(z_raw[0])

        # 3. 先得到局部 ENU 坐标
        x_local = np.zeros_like(t_pts)
        y_local = np.zeros_like(t_pts)

        for i in range(len(t_pts)):
            x_local[i], y_local[i] = self.gps_rad_to_local_xy(
                float(lat_pts[i]),
                float(lon_pts[i])
            )

        # 4. 计算 CPS 轨迹首段方向
        # dx0 = x_local[min(1, len(x_local)-1)] - x_local[0]
        # dy0 = y_local[min(1, len(y_local)-1)] - y_local[0]
        # gps_yaw0 = math.atan2(dy0, dx0)
        gps_yaw0 = 0.0
        found_heading = False
        for j in range(1, len(x_local)):
            dx0 = x_local[j] - x_local[0]
            dy0 = y_local[j] - y_local[0]
            if math.hypot(dx0, dy0) > 0.05:
                gps_yaw0 = math.atan2(dy0, dx0)
                found_heading = True
                break

        if not found_heading:
            self.get_logger().warn("CPS轨迹起始段位移过小，初始方向暂用车辆当前航向")
            gps_yaw0 = self.theta_act

        # 5. 把 CPS 轨迹起点对齐到当前 odom 车辆位置
        self.cps_odom_x0 = float(self.x_act)
        self.cps_odom_y0 = float(self.y_act)

        # 6. 把 GPS 首段方向旋转到车辆当前 odom 航向
        self.cps_odom_yaw0 = float(self.theta_act - gps_yaw0)

        # 7. ENU 坐标转 odom 坐标
        x_pts = np.zeros_like(t_pts)
        y_pts = np.zeros_like(t_pts)

        for i in range(len(t_pts)):
            x_pts[i], y_pts[i] = self.local_xy_to_odom_xy(
                float(x_local[i]),
                float(y_local[i])
            )

        z_pts = z_raw - self.cps_alt0

        # 8. 轨迹航向角：由 odom 下的 x/y 计算
        dx_grad = np.gradient(x_pts)
        dy_grad = np.gradient(y_pts)
        theta_pts = np.unwrap(np.arctan2(dy_grad, dx_grad))

        # 9. 速度：直接使用 CPS 文件速度列
        v_pts = v_kmh / 3.6
        v_pts = np.clip(v_pts, self.v_min, self.v_max)

        # 10. 角速度：由航向角对时间差分
        dt = np.diff(t_pts)
        dtheta = np.diff(theta_pts)
        omega_pts = np.zeros_like(t_pts)
        omega_pts[1:] = dtheta / np.maximum(dt, 1e-6)
        omega_pts[0] = omega_pts[1]
        omega_pts = np.clip(omega_pts, self.omega_min, self.omega_max)

        # 11. 坡度
        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        dz = np.diff(z_pts)
        ds = np.sqrt(dx**2 + dy**2)

        alpha_pts = np.zeros_like(t_pts)
        alpha_pts[1:] = np.arctan2(dz, np.maximum(ds, 1e-6))
        alpha_pts[0] = alpha_pts[1]
        alpha_pts = np.clip(alpha_pts, -self.max_slope_rad, self.max_slope_rad)

        # 12. 横坡暂时设 0
        beta_pts = np.full_like(t_pts, self.beta_default, dtype=float)
        beta_pts = np.clip(beta_pts, -self.max_beta_rad, self.max_beta_rad)

        # 13. 附着系数暂时用默认值
        mu_pts = np.full_like(t_pts, self.mu_nominal, dtype=float)
        mu_pts = np.clip(mu_pts, self.mu_min, self.mu_max)

        # 14. 建立插值函数，后面的 control_loop 可以继续使用
        self.t_track = t_pts
        self.t_final = float(t_pts[-1])

        # 自动生成检查点时间：每 1 秒一个检查点，不包括起点
        # 你也可以改成 [2.0, 4.0, 6.0, self.t_final]
        if self.use_checkpoint_soft_constraint:
            cp_dt = 1.0
            cp_start = 2.0    
            self.checkpoint_times = list(np.arange(cp_start, self.t_final + 1e-6, cp_dt))

            # 确保终点也作为检查点
            if len(self.checkpoint_times) == 0 or abs(self.checkpoint_times[-1] - self.t_final) > 1e-3:
                self.checkpoint_times.append(self.t_final)

            self.cp_arrival_logged = {float(t): None for t in self.checkpoint_times}
        else:
            self.checkpoint_times = []
            self.cp_arrival_logged = {}

        self.traj_interp = {
            'x_of_t': interp1d(t_pts, x_pts, bounds_error=False, fill_value=(x_pts[0], x_pts[-1])),
            'y_of_t': interp1d(t_pts, y_pts, bounds_error=False, fill_value=(y_pts[0], y_pts[-1])),
            'theta_of_t': interp1d(t_pts, theta_pts, bounds_error=False, fill_value=(theta_pts[0], theta_pts[-1])),
            'v_of_t': interp1d(t_pts, v_pts, bounds_error=False, fill_value=(v_pts[0], v_pts[-1])),
            'omega_of_t': interp1d(t_pts, omega_pts, bounds_error=False, fill_value=(omega_pts[0], omega_pts[-1])),
            'alpha_of_t': interp1d(t_pts, alpha_pts, bounds_error=False, fill_value=(alpha_pts[0], alpha_pts[-1])),
            'beta_of_t': interp1d(t_pts, beta_pts, bounds_error=False, fill_value=(beta_pts[0], beta_pts[-1])),
            'mu_of_t': interp1d(t_pts, mu_pts, bounds_error=False, fill_value=(mu_pts[0], mu_pts[-1])),
        }

        if self.use_checkpoint_soft_constraint and self.checkpoint_times is not None:
            self.get_logger().info("检查点列表如下：")
            for t_cp in self.checkpoint_times:
                x_cp = float(self.traj_interp['x_of_t'](t_cp))
                y_cp = float(self.traj_interp['y_of_t'](t_cp))
                self.get_logger().info(
                    f"CP: t={t_cp:.3f}s, x={x_cp:.3f}m, y={y_cp:.3f}m"
                )

        self.path_ready = True
        self.is_initialized = False
        self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)

        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = "odom"
        self.arrival_time = None
        self.total_ctrl_count = 0
        self.v_sat_count = 0
        self.w_sat_count = 0

        self.publish_cps_reference_path_for_rviz(x_pts, y_pts, theta_pts, t_pts)
        self.publish_checkpoint_markers_for_rviz()

        self.get_logger().info(
            f"CPS轨迹已转换: N={len(t_pts)}, "
            f"t_final={self.t_final:.3f}s, "
            f"v_min={np.min(v_pts):.3f}m/s, v_max={np.max(v_pts):.3f}m/s"
        )

    def control_loop(self):
        if not self.path_ready or not self.odom_ready:
            return

        if not self.is_initialized:
            dists = np.sqrt(
                (self.traj_interp['x_of_t'](self.t_track) - self.x_act)**2 +
                (self.traj_interp['y_of_t'](self.t_track) - self.y_act)**2
            )
            idx0 = np.argmin(dists)
            self.t_ref_offset = float(self.t_track[idx0])
            self.start_time = self.get_clock().now()
            self.u_prev = np.array([self.v_act, self.omega_act])
            self.is_initialized = True
            self.get_logger().info(f"初始对齐完毕, idx0 = {idx0}, t0 = {self.t_ref_offset:.2f} s")
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        t_ref_now = float(np.clip(self.t_ref_offset + elapsed, 0.0, self.t_final))

        x_ref_k = float(self.traj_interp['x_of_t'](t_ref_now))
        y_ref_k = float(self.traj_interp['y_of_t'](t_ref_now))
        theta_ref_k = float(self.traj_interp['theta_of_t'](t_ref_now))

        dx = self.x_act - x_ref_k
        dy = self.y_act - y_ref_k
        delta_p = dx * math.cos(theta_ref_k) + dy * math.sin(theta_ref_k)

        e_t = delta_p
        e_n = -dx * math.sin(theta_ref_k) + dy * math.cos(theta_ref_k)
        e_theta = math.atan2(
            math.sin(self.theta_act - theta_ref_k),
            math.cos(self.theta_act - theta_ref_k)
        )

        v_ref_now = float(self.traj_interp['v_of_t'](t_ref_now))
        time_err = e_t / max(abs(v_ref_now), self.ll_v_eps)
        time_err = float(np.clip(time_err, -self.ll_time_err_max, self.ll_time_err_max))
        dv_lag_base = -self.ll_K_time_v * time_err
        dv_lag_base = float(np.clip(dv_lag_base, -self.ll_dv_max, self.ll_dv_max))

        goal_x = float(self.traj_interp['x_of_t'](self.t_final))
        goal_y = float(self.traj_interp['y_of_t'](self.t_final))
        goal_dist = math.hypot(self.x_act - goal_x, self.y_act - goal_y)

        if self.arrival_time is None and goal_dist < 0.10:
            self.arrival_time = elapsed
            self.get_logger().info(
                f"首次进入终点邻域: t_arrive={self.arrival_time:.3f}s, "
                f"t_ref_final={self.t_final:.3f}s, "
                f"误差={self.arrival_time - self.t_final:.3f}s"
            )

        if t_ref_now >= self.t_final and goal_dist < 0.10:
            self.stop_robot()
            self.get_logger().info("按计划到达终点！", once=True)
            return

        X_ref_mod = np.zeros((self.nx, self.Np + 1))
        U_ref_mod = np.zeros((self.nu, self.Np))
        Terrain_ref_mod = np.zeros((self.np_terrain, self.Np))

        # 新增：检查点参数
        CP_ref_mod = np.zeros((3, self.Np + 1))

        for i in range(self.Np + 1):
            t_i = min(t_ref_now + i * self.dt, self.t_final)

            x_ref = float(self.traj_interp['x_of_t'](t_i))
            y_ref = float(self.traj_interp['y_of_t'](t_i))
            theta_ref = float(self.traj_interp['theta_of_t'](t_i))
            v_ref = float(self.traj_interp['v_of_t'](t_i))
            omega_ref = float(self.traj_interp['omega_of_t'](t_i))
            alpha_ref = float(self.traj_interp['alpha_of_t'](t_i))
            beta_ref = float(self.traj_interp['beta_of_t'](t_i))
            mu_ref = float(self.traj_interp['mu_of_t'](t_i))

            ll_decay = math.exp(-0.5 * i * self.dt)
            dv_lag_i = dv_lag_base * ll_decay
            v_ref_ll = v_ref + dv_lag_i
            v_ref_ll = float(np.clip(v_ref_ll, self.v_min, self.v_max))
            kappa_ref = omega_ref / max(abs(v_ref), self.ll_v_eps)
            omega_ref_ll = kappa_ref * v_ref_ll
            omega_ref_ll = float(np.clip(omega_ref_ll, self.omega_min, self.omega_max))

            X_ref_mod[0, i] = x_ref
            X_ref_mod[1, i] = y_ref
            X_ref_mod[2, i] = theta_ref
            X_ref_mod[3, i] = v_ref_ll
            X_ref_mod[4, i] = 0.0
            X_ref_mod[5, i] = omega_ref_ll

            if i < self.Np:
                t_next = min(t_i + self.dt, self.t_final)

                v_next = float(self.traj_interp['v_of_t'](t_next))
                omega_next = float(self.traj_interp['omega_of_t'](t_next))

                ll_decay_next = math.exp(-0.5 * (i + 1) * self.dt)
                dv_lag_next = dv_lag_base * ll_decay_next

                v_next_ll = v_next + dv_lag_next
                v_next_ll = float(np.clip(v_next_ll, self.v_min, self.v_max))

                kappa_next = omega_next / max(abs(v_next), self.ll_v_eps)
                omega_next_ll = kappa_next * v_next_ll
                omega_next_ll = float(np.clip(omega_next_ll, self.omega_min, self.omega_max))

                a_ref_ll = (v_next_ll - v_ref_ll) / self.dt
                r_dot_ref_ll = (omega_next_ll - omega_ref_ll) / self.dt

                if self.use_slope_feedforward:
                    slope_ff = self.ff_slope_gain * self.g_const * math.sin(alpha_ref)
                else:
                    slope_ff = 0.0

                v_cmd_ff = (
                    v_ref_ll
                    + self.tau_v * (a_ref_ll + slope_ff)
                ) / max(self.kv_nominal, 1e-6)

                omega_cmd_ff = (
                    omega_ref_ll
                    + self.tau_r * r_dot_ref_ll
                ) / max(self.kr_nominal, 1e-6)

                U_ref_mod[0, i] = np.clip(v_cmd_ff, self.v_min, self.v_max)
                U_ref_mod[1, i] = np.clip(omega_cmd_ff, self.omega_min, self.omega_max)

                Terrain_ref_mod[0, i] = np.clip(alpha_ref, -self.max_slope_rad, self.max_slope_rad)
                Terrain_ref_mod[1, i] = np.clip(beta_ref, -self.max_beta_rad, self.max_beta_rad)
                Terrain_ref_mod[2, i] = np.clip(mu_ref, self.mu_min, self.mu_max)

        # 将落入当前预测域的检查点写入 CP_ref_mod
        if self.use_checkpoint_soft_constraint and self.checkpoint_times is not None:
            horizon_start = t_ref_now
            horizon_end = t_ref_now + self.Np * self.dt

            for t_cp in self.checkpoint_times:
                t_cp = float(t_cp)

                # 检查点不在当前预测域内，不施加约束
                if t_cp < horizon_start or t_cp > horizon_end:
                    continue

                k_cp = int(round((t_cp - t_ref_now) / self.dt))
                k_cp = int(np.clip(k_cp, 0, self.Np))

                x_cp = float(self.traj_interp['x_of_t'](t_cp))
                y_cp = float(self.traj_interp['y_of_t'](t_cp))

                CP_ref_mod[0, k_cp] = x_cp
                CP_ref_mod[1, k_cp] = y_cp
                CP_ref_mod[2, k_cp] = self.cp_weight

        theta_for_mpc = self.theta_act
        while theta_for_mpc - theta_ref_k > math.pi:
            theta_for_mpc -= 2 * math.pi
        while theta_for_mpc - theta_ref_k < -math.pi:
            theta_for_mpc += 2 * math.pi

        x0_dyn = np.array([
            self.x_act,
            self.y_act,
            theta_for_mpc,
            np.clip(self.v_act, self.v_min, self.v_max),
            np.clip(self.vy_act, self.vy_min, self.vy_max),
            np.clip(self.omega_act, self.omega_min, self.omega_max)
        ])

        opt_params_k = np.concatenate([
            X_ref_mod.flatten(order='F'),
            U_ref_mod.flatten(order='F'),
            Terrain_ref_mod.flatten(order='F'),
            CP_ref_mod.flatten(order='F'),
            x0_dyn,
            self.u_prev
        ])

        try:
            tic = time.perf_counter()
            res = self.solver(
                x0=self.u0_guess,
                p=opt_params_k,
                lbg=self.lbg,
                ubg=self.ubg,
                lbx=self.lb_opt,
                ubx=self.ub_opt
            )
            solve_ms = (time.perf_counter() - tic) * 1000.0
        except RuntimeError as e:
            self.get_logger().error(f"NMPC求解失败: {e}")
            self.stop_robot()
            return

        stats = self.solver.stats()
        if not stats.get('success', False):
            self.get_logger().error(
                f"NMPC求解未成功: {stats.get('return_status', 'unknown')}"
            )
            self.stop_robot()
            return

        opt_sol = np.array(res['x']).flatten()
        self.u0_guess = opt_sol

        u_opt = opt_sol[self.nx * (self.Np + 1):].reshape(self.nu, self.Np, order='F')
        v_cmd = float(u_opt[0, 0])
        omega_cmd = float(u_opt[1, 0])

        self.total_ctrl_count += 1
        if abs(v_cmd - self.v_max) < 1e-3 or abs(v_cmd - self.v_min) < 1e-3:
            self.v_sat_count += 1
        if abs(omega_cmd - self.omega_max) < 1e-3 or abs(omega_cmd - self.omega_min) < 1e-3:
            self.w_sat_count += 1

        v_sat_ratio = self.v_sat_count / max(self.total_ctrl_count, 1)
        w_sat_ratio = self.w_sat_count / max(self.total_ctrl_count, 1)

        self.u_prev = np.array([v_cmd, omega_cmd])

        cmd_msg = Twist()
        cmd_msg.linear.x = v_cmd
        cmd_msg.angular.z = omega_cmd
        self.cmd_pub.publish(cmd_msg)

        self.publish_scalar(self.et_pub, e_t)
        self.publish_scalar(self.en_pub, e_n)
        self.publish_scalar(self.eth_pub, e_theta)
        self.publish_scalar(self.tref_pub, t_ref_now)
        self.publish_scalar(self.goal_dist_pub, goal_dist)
        self.publish_scalar(self.solve_ms_pub, solve_ms)
        self.publish_scalar(self.vcmd_pub, v_cmd)
        self.publish_scalar(self.wcmd_pub, omega_cmd)
        self.publish_scalar(self.v_sat_ratio_pub, v_sat_ratio)
        self.publish_scalar(self.w_sat_ratio_pub, w_sat_ratio)

        # 记录检查点实际通过时间
        if self.use_checkpoint_soft_constraint and self.checkpoint_times is not None:
            for t_cp in self.checkpoint_times:
                t_cp = float(t_cp)

                if self.cp_arrival_logged.get(t_cp, None) is not None:
                    continue

                x_cp = float(self.traj_interp['x_of_t'](t_cp))
                y_cp = float(self.traj_interp['y_of_t'](t_cp))
                dist_cp = math.hypot(self.x_act - x_cp, self.y_act - y_cp)

                if dist_cp < self.cp_arrival_radius:
                    actual_ref_time = self.t_ref_offset + elapsed
                    self.cp_arrival_logged[t_cp] = actual_ref_time
                    self.get_logger().info(
                        f"检查点通过: t_cp_ref={t_cp:.3f}s, "
                        f"t_arrive_ref={actual_ref_time:.3f}s, "
                        f"time_error={actual_ref_time - t_cp:.3f}s, "
                        f"dist={dist_cp:.3f}m"
                    )

        alpha_now = float(self.traj_interp['alpha_of_t'](t_ref_now))
        beta_now = float(self.traj_interp['beta_of_t'](t_ref_now))
        mu_now = float(self.traj_interp['mu_of_t'](t_ref_now))
        self.get_logger().info(
            f"Δp: {delta_p:.3f} m | time_err: {time_err:.3f} s | dv_lag: {dv_lag_base:.3f} | "
            f"t_ref: {t_ref_now:.2f}/{self.t_final:.2f} s | "
            f"alpha: {alpha_now:.3f} rad | beta: {beta_now:.3f} rad | mu: {mu_now:.2f} | "
            f"v_act: {self.v_act:.2f} | vy_act: {self.vy_act:.2f} | r_act: {self.omega_act:.2f} | "
            f"cmd_v: {v_cmd:.2f} | cmd_w: {omega_cmd:.2f} | solve={solve_ms:.1f} ms"
        )
        self.publish_scalar(self.alpha_pub, alpha_now)
        self.publish_scalar(self.beta_pub, beta_now)
        self.publish_scalar(self.mu_pub, mu_now)
        self.publish_scalar(self.vy_pub, self.vy_act)
        
        self.publish_mpc_trajectory(opt_sol)
        self.publish_actual_path()

    def publish_mpc_trajectory(self, opt_sol):
        X_opt = opt_sol[:self.nx * (self.Np + 1)].reshape(self.nx, self.Np + 1, order='F')

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"

        for i in range(self.Np + 1):
            pose = PoseStamped()
            pose.pose.position.x = float(X_opt[0, i])
            pose.pose.position.y = float(X_opt[1, i])
            path_msg.poses.append(pose)

        self.mpc_path_pub.publish(path_msg)

    def publish_actual_path(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = float(self.x_act)
        pose.pose.position.y = float(self.y_act)

        self.actual_path_msg.header.stamp = pose.header.stamp
        self.actual_path_msg.poses.append(pose)

        max_len = 2000
        if len(self.actual_path_msg.poses) > max_len:
            self.actual_path_msg.poses = self.actual_path_msg.poses[-max_len:]

        self.actual_path_pub.publish(self.actual_path_msg)

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NMPCLeadLagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

