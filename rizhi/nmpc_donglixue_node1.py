import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64
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
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/global_path', self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc_path', 10) 
        self.actual_path_pub = self.create_publisher(Path, '/actual_path', 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.x_act = 0.0
        self.y_act = 0.0
        self.psi_act = 0.0
        
        self.u_body_act = 0.0
        self.v_body_act = 0.0
        self.r_act = 0.0
        
        self.path_ready = False  
        self.odom_ready = False      
        self.t_track = None            # 参考时间轴
        self.t_final = 0.0             # 轨迹总时间
        self.start_time = None         # 控制器开始计时时间
        self.t_ref_offset = 0.0        # 初始对齐时的时间偏置 
        self.is_initialized = False 

        self.cmd_prev = np.array([0.0, 0.0])

######灰箱模型参数，待定
        self.tau_u = 0.25
        self.tau_r = 0.18
        self.k_vr = 0.55
        self.c_v = 1.20
        self.c_r =0.35

        self.d_hat = np.zeros(3)

        self.total_ctrl_count = 0
        self.u_sat_count = 0
        self.r_sat_count = 0
        self.arrival_time = None

        self.et_pub = self.create_publisher(Float64, '/nmpc_debug/e_t', 10)
        self.en_pub = self.create_publisher(Float64, '/nmpc_debug/e_n', 10)
        self.epsi_pub = self.create_publisher(Float64, '/nmpc_debug/e_psi', 10)
        self.tref_pub = self.create_publisher(Float64, '/nmpc_debug/t_ref', 10)
        self.goal_dist_pub = self.create_publisher(Float64, '/nmpc_debug/goal_dist', 10)
        self.solve_ms_pub = self.create_publisher(Float64, '/nmpc_debug/solve_ms', 10)
        self.ucmd_pub = self.create_publisher(Float64, '/nmpc_debug/u_cmd', 10)
        self.rcmd_pub = self.create_publisher(Float64, '/nmpc_debug/r_cmd', 10)
        self.u_sat_ratio_pub = self.create_publisher(Float64, '/nmpc_debug/u_sat_ratio', 10)
        self.r_sat_ratio_pub = self.create_publisher(Float64, '/nmpc_debug/r_sat_ratio', 10)
        self.u_body_pub = self.create_publisher(Float64, '/nmpc_debug/u_body', 10)
        self.v_body_pub = self.create_publisher(Float64, '/nmpc_debug/v_body', 10)
        self.r_body_pub = self.create_publisher(Float64, '/nmpc_debug/r_body', 10)

        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = 'odom'
        
        self.get_logger().info("初始化 CasADi NMPC 求解器...")
        self.init_casadi_solver()
        self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)
        self.get_logger().info(" NMPC Lead-Lag 节点已启动！等待接收全局路径...")


    def publish_scalar(self, pub, value: float):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    def wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def init_casadi_solver(self):
        self.Np = 20
        self.nx = 6
        self.nu = 2

        self.u_cmd_min, self.u_cmd_max = -0.5, 2.0
        self.r_cmd_min, self.r_cmd_max = -math.pi/2, math.pi/2

        x = ca.SX.sym('x', self.nx)
        cmd = ca.SX.sym('cmd', self.nu)
        d_hat = ca.SX.sym('d_hat', 3)

        px, py, psi, u_b, v_b, r_b = x[0], x[1], x[2], x[3], x[4], x[5]
        u_cmd, r_cmd = cmd[0], cmd[1]

        px_dot = u_b * ca.cos(psi) - v_b * ca.sin(psi)
        py_dot = u_b * ca.sin(psi) + v_b * ca.cos(psi)
        psi_dot = r_b

        u_dot = r_b * v_b + (u_cmd - u_b) / self.tau_u - d_hat[0]
        v_dot = -r_b * u_b - self.c_v * v_b + self.k_vr * r_cmd - d_hat[1]
        r_dot = (r_cmd - r_b) / self.tau_r - self.c_r * v_b - d_hat[2]

        f_continuous = ca.Function(
            'f_continuous',
            [x, cmd, d_hat],
            [ca.vertcat(px_dot, py_dot, psi_dot, u_dot, v_dot, r_dot)]
        )

        k1 = f_continuous(x, cmd, d_hat)
        k2 = f_continuous(x + self.dt/2 * k1, cmd, d_hat)
        k3 = f_continuous(x + self.dt/2 * k2, cmd, d_hat)
        k4 = f_continuous(x + self.dt * k3, cmd, d_hat)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        F_discrete = ca.Function('F_discrete', [x, cmd, d_hat], [x_next])

        X = ca.SX.sym('X', self.nx, self.Np+1)
        U = ca.SX.sym('U', self.nu, self.Np)
        X_ref = ca.SX.sym('X_ref', self.nx, self.Np+1)
        U_ref = ca.SX.sym('U_ref', self.nu, self.Np)
        x0 = ca.SX.sym('x0', self.nx)
        cmd_prev = ca.SX.sym('cmd_prev', self.nu)
        d_param = ca.SX.sym('d_param', 3)

        J = 0
        g = [X[:, 0] - x0]

        Qdu = ca.DM(np.diag([1.5, 1.0]))

        for k in range(self.Np):
            dx = X[0, k] - X_ref[0, k]
            dy = X[1, k] - X_ref[1, k]
            psi_ref = X_ref[2, k]

###每个预测步都要计算偏差
            e_t = dx * ca.cos(psi_ref) + dy * ca.sin(psi_ref)
            e_n = -dx * ca.sin(psi_ref) + dy * ca.cos(psi_ref)
            e_psi = ca.atan2(ca.sin(X[2, k] - psi_ref), ca.cos(X[2, k] - psi_ref))

            e_u = X[3, k] - X_ref[3, k]
            e_v = X[4, k] - X_ref[4, k]
            e_r = X[5, k] - X_ref[5, k]

            e_ucmd = U[0, k] - U_ref[0, k]
            e_rcmd = U[1, k] - U_ref[1, k]

            dcmd = U[:, k] - (cmd_prev if k == 0 else U[:, k-1])

###加入代价函数       待定
            J += 40.0 * e_t**2 + 80.0 * e_n**2 + 8.0 * e_psi**2   
            J += 10.0 * e_u**2 + 4.0 * e_v**2 + 6.0 * e_r**2
            J += 0.8 * e_ucmd**2 + 0.5 * e_rcmd**2
            J += ca.mtimes([dcmd.T, Qdu, dcmd])

            g.append(X[:, k+1] - F_discrete(X[:, k], U[:, k], d_param))

        dxN = X[0, self.Np] - X_ref[0, self.Np]
        dyN = X[1, self.Np] - X_ref[1, self.Np]
        psi_refN = X_ref[2, self.Np]

        e_tN = dxN * ca.cos(psi_refN) + dyN * ca.sin(psi_refN)
        e_nN = -dxN * ca.sin(psi_refN) + dyN * ca.cos(psi_refN)
        e_psiN = ca.atan2(ca.sin(X[2, self.Np] - psi_refN), ca.cos(X[2, self.Np] - psi_refN))

        e_uN = X[3, self.Np] - X_ref[3, self.Np]
        e_vN = X[4, self.Np] - X_ref[4, self.Np]
        e_rN = X[5, self.Np] - X_ref[5, self.Np]

####不仅在当前要加入时间代价，在预测步长里也要加上      待定
        J += 120.0 * e_tN**2 + 150.0 * e_nN**2 + 12.0 * e_psiN**2
        J += 18.0 * e_uN**2 + 8.0 * e_vN**2 + 10.0 * e_rN**2

        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = ca.vertcat(
            ca.reshape(X_ref, -1, 1),
            ca.reshape(U_ref, -1, 1),
            x0,
            cmd_prev,
            d_param
        )

        nlp = {'x': opt_variables, 'p': opt_params, 'f': J, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 200}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbg = np.zeros(self.nx * (self.Np + 1))
        self.ubg = np.zeros(self.nx * (self.Np + 1))

        lbx = np.full(self.nx * (self.Np + 1), -np.inf)
        ubx = np.full(self.nx * (self.Np + 1), np.inf)

        for i in range(self.Np + 1):
            base = i * self.nx
            lbx[base + 3] = -0.8
            ubx[base + 3] =  2.5
            lbx[base + 4] = -1.5
            ubx[base + 4] =  1.5
            lbx[base + 5] = -2.0
            ubx[base + 5] =  2.0
        
        lbu = np.tile([self.u_cmd_min, self.r_cmd_min], self.Np)
        ubu = np.tile([self.u_cmd_max, self.r_cmd_max], self.Np)

        self.lb_opt = np.concatenate([lbx, lbu])
        self.ub_opt = np.concatenate([ubx, ubu])

    def odom_callback(self, msg):
        self.x_act = msg.pose.pose.position.x
        self.y_act = msg.pose.pose.position.y
        # 从四元数提取航向角 yaw
        q = msg.pose.pose.orientation
        self.psi_act = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.u_body_act = msg.twist.twist.linear.x
        self.v_body_act = msg.twist.twist.linear.y
        self.r_act = msg.twist.twist.angular.z
        self.odom_ready = True

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            self.get_logger().warn('路径点不足，忽略本次路径')
            return
###从得到的轨迹中提取时间信息
        self.get_logger().info("收到带时间戳参考轨迹，直接建立时间插值...")

        x_pts, y_pts, psi_pts, t_pts = [], [], [], []
        t0_sec = None
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            
            q = pose_stamped.pose.orientation
            psi = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            sec = pose_stamped.header.stamp.sec + pose_stamped.header.stamp.nanosec * 1e-9
            if t0_sec is None:
                t0_sec = sec
            t_rel = sec - t0_sec

            x_pts.append(x)
            y_pts.append(y)
            psi_pts.append(psi)
            t_pts.append(t_rel)

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        psi_pts = np.unwrap(np.array(psi_pts))
        t_pts = np.array(t_pts)

        keep = np.hstack(([True], np.diff(t_pts) > 1e-6))
        x_pts = x_pts[keep]
        y_pts = y_pts[keep]
        psi_pts = psi_pts[keep]
        t_pts = t_pts[keep]

#####读取轨迹得到位置，姿态，时间戳信息

        if len(x_pts) < 2:
            self.get_logger().warn("路径有效点不足，忽略本次路径")
            return

        dt = np.diff(t_pts)
        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        dpsi = np.diff(psi_pts)

#####差分估计得到v和omega
        u_ref_pts = np.zeros_like(t_pts)
        v_ref_pts = np.zeros_like(t_pts)   # 先设为 0
        r_ref_pts = np.zeros_like(t_pts)

        u_ref_pts[1:] = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        u_ref_pts[0] = u_ref_pts[1]

        r_ref_pts[1:] = dpsi / np.maximum(dt, 1e-6)
        r_ref_pts[0] = r_ref_pts[1]

        # 建议限幅，避免参考本身太激进
        u_ref_pts = np.clip(u_ref_pts, self.u_cmd_min, self.u_cmd_max)
        r_ref_pts = np.clip(r_ref_pts, self.r_cmd_min, self.r_cmd_max)

        self.t_track = t_pts
        self.t_final = float(t_pts[-1])

####从轨迹中得到的时间关联参数
        self.traj_interp = {
            'x_of_t': interp1d(t_pts, x_pts, bounds_error=False, fill_value=(x_pts[0], x_pts[-1])),
            'y_of_t': interp1d(t_pts, y_pts, bounds_error=False, fill_value=(y_pts[0], y_pts[-1])),
            'psi_of_t': interp1d(t_pts, psi_pts, bounds_error=False, fill_value=(psi_pts[0], psi_pts[-1])),
            'u_of_t': interp1d(t_pts, u_ref_pts, bounds_error=False, fill_value=(u_ref_pts[0], u_ref_pts[-1])),
            'v_of_t': interp1d(t_pts, v_ref_pts, bounds_error=False, fill_value=(v_ref_pts[0], v_ref_pts[-1])),
            'r_of_t': interp1d(t_pts, r_ref_pts, bounds_error=False, fill_value=(r_ref_pts[0], r_ref_pts[-1])),
        }

        self.path_ready = True
        self.is_initialized = False

        self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)
        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = "odom"
        self.arrival_time = None
        self.total_ctrl_count = 0
        self.u_sat_count = 0
        self.r_sat_count = 0

    def shift_guess(self, opt_sol: np.ndarray) -> np.ndarray:
        X = opt_sol[:self.nx * (self.Np + 1)].reshape(self.nx, self.Np + 1, order='F')
        U = opt_sol[self.nx * (self.Np + 1):].reshape(self.nu, self.Np, order='F')

        X_shift = np.hstack([X[:, 1:], X[:, -1:]])
        U_shift = np.hstack([U[:, 1:], U[:, -1:]])

        return np.concatenate([
            X_shift.flatten(order='F'),
            U_shift.flatten(order='F')
        ])

    def control_loop(self):
        if not self.path_ready or not self.odom_ready:
            return
#####找到小车当前位置离哪个参考时刻的位置最近，将当前状态与参考轨迹时间进度对齐
        if not self.is_initialized:
            x_samples = np.asarray(self.traj_interp['x_of_t'](self.t_track), dtype=float)
            y_samples = np.asarray(self.traj_interp['y_of_t'](self.t_track), dtype=float)
            dists = np.sqrt((x_samples - self.x_act) ** 2 + (y_samples - self.y_act) ** 2)
            idx0 = int(np.argmin(dists))

            self.t_ref_offset = float(self.t_track[idx0])
            self.start_time = self.get_clock().now()
            self.cmd_prev = np.array([self.u_body_act, self.r_act])
            self.is_initialized = True
            self.get_logger().info(
                f'初始对齐完毕: idx0={idx0}, t_ref_offset={self.t_ref_offset:.2f}s'
            )
            return

####参考时间是初始对齐时间加上时间流逝，从而根据这个时间确定参考点
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        t_ref_now = float(np.clip(self.t_ref_offset + elapsed, 0.0, self.t_final))

###确定按照时间计划在的位置角度
        x_ref_k = float(self.traj_interp['x_of_t'](t_ref_now))
        y_ref_k = float(self.traj_interp['y_of_t'](t_ref_now))
        psi_ref_k = float(self.traj_interp['psi_of_t'](t_ref_now))

        dx = self.x_act - x_ref_k
        dy = self.y_act - y_ref_k
        e_t = dx * math.cos(psi_ref_k) + dy * math.sin(psi_ref_k)
        e_n = -dx * math.sin(psi_ref_k) + dy * math.cos(psi_ref_k)
        e_psi = self.wrap_angle(self.psi_act - psi_ref_k)

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

        for i in range(self.Np + 1):
            t_i = min(t_ref_now + i * self.dt, self.t_final)

            X_ref_mod[0, i] = float(self.traj_interp['x_of_t'](t_i))
            X_ref_mod[1, i] = float(self.traj_interp['y_of_t'](t_i))
            X_ref_mod[2, i] = float(self.traj_interp['psi_of_t'](t_i))
            X_ref_mod[3, i] = float(self.traj_interp['u_of_t'](t_i))
            X_ref_mod[4, i] = float(self.traj_interp['v_of_t'](t_i))   # 一开始基本就是 0
            X_ref_mod[5, i] = float(self.traj_interp['r_of_t'](t_i))

            if i < self.Np:
                U_ref_mod[0, i] = X_ref_mod[3, i]
                U_ref_mod[1, i] = X_ref_mod[5, i]

        psi_for_mpc = self.psi_act
        while psi_for_mpc - X_ref_mod[2, 0] > math.pi:
            psi_for_mpc -= 2.0 * math.pi
        while psi_for_mpc - X_ref_mod[2, 0] < -math.pi:
            psi_for_mpc += 2.0 * math.pi

        x0_mpc = np.array([
            self.x_act,
            self.y_act,
            psi_for_mpc,
            self.u_body_act,
            self.v_body_act,
            self.r_act
        ])

        opt_params_k = np.concatenate([
            X_ref_mod.flatten(order='F'),
            U_ref_mod.flatten(order='F'),
            x0_mpc,
            self.cmd_prev,
            self.d_hat
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
        self.u0_guess = self.shift_guess(opt_sol)

        u_opt = opt_sol[self.nx * (self.Np + 1):].reshape(self.nu, self.Np, order='F')
        u_cmd = float(u_opt[0, 0])
        r_cmd = float(u_opt[1, 0])

        self.total_ctrl_count += 1
        if abs(u_cmd - self.u_cmd_max) < 1e-3 or abs(u_cmd - self.u_cmd_min) < 1e-3:
            self.u_sat_count += 1
        if abs(r_cmd - self.r_cmd_max) < 1e-3 or abs(r_cmd - self.r_cmd_min) < 1e-3:
            self.r_sat_count += 1

        u_sat_ratio = self.u_sat_count / max(self.total_ctrl_count, 1)
        r_sat_ratio = self.r_sat_count / max(self.total_ctrl_count, 1)

        self.cmd_prev = np.array([u_cmd, r_cmd])

        cmd_msg = Twist()
        cmd_msg.linear.x = u_cmd
        cmd_msg.angular.z = r_cmd
        self.cmd_pub.publish(cmd_msg)

        self.publish_scalar(self.et_pub, e_t)
        self.publish_scalar(self.en_pub, e_n)
        self.publish_scalar(self.epsi_pub, e_psi)
        self.publish_scalar(self.tref_pub, t_ref_now)
        self.publish_scalar(self.goal_dist_pub, goal_dist)
        self.publish_scalar(self.solve_ms_pub, solve_ms)
        self.publish_scalar(self.ucmd_pub, u_cmd)
        self.publish_scalar(self.rcmd_pub, r_cmd)
        self.publish_scalar(self.u_sat_ratio_pub, u_sat_ratio)
        self.publish_scalar(self.r_sat_ratio_pub, r_sat_ratio)
        self.publish_scalar(self.u_body_pub, self.u_body_act)
        self.publish_scalar(self.v_body_pub, self.v_body_act)
        self.publish_scalar(self.r_body_pub, self.r_act)

        self.get_logger().info(
            f'e_t={e_t:.3f} m | e_n={e_n:.3f} m | t_ref={t_ref_now:.2f}/{self.t_final:.2f} s | '
            f'u_cmd={u_cmd:.2f} | r_cmd={r_cmd:.2f} | solve={solve_ms:.1f} ms'
        )

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