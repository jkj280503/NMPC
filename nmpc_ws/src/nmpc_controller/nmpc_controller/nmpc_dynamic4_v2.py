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
        self.last_path_signature = None

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/global_path', self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc_path', 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.x_act = 0.0
        self.y_act = 0.0
        self.theta_act = 0.0
        self.v_act = 0.0
        self.vy_act = 0.0
        self.omega_act = 0.0

        # 版本4：6状态 + 纵坡 + 横坡 + 物理灰箱动力学模型参数
        # 状态: [X, Y, theta, vx, vy, r]
        # 输入: [v_cmd, omega_cmd]，仍然保持 /cmd_vel 接口不变。
        # 预瞄地形参数: alpha, beta, mu
        # alpha > 0 表示沿参考轨迹前进方向上坡；beta 表示路径法向横坡角；mu 为名义附着系数。
        #
        # 版本4相对版本3的核心变化：
        # 1) vx_dot 不再直接使用一阶速度响应，而是写成刚体纵向动力学：
        #    vx_dot = vy*r + (Fx - F_roll - F_drag)/m - slope_gain*g*sin(alpha)
        # 2) vy_dot 写成横向动力学：
        #    vy_dot = -vx*r + Fy/m - cross_slope_gain*g*sin(beta)
        # 3) r_dot 写成横摆动力学：
        #    r_dot = (Mz - M_res)/Iz
        # 4) Fx 和 Mz 来自 /cmd_vel 等效力/等效力矩，并受 mu*m*g*cos(gamma) 附着上限约束。

        # ---------- linorobot2 4wd 仿真模型中提取的物理参数 ----------
        # base_mass = 10 kg, wheel_mass = 0.6 kg, 4个轮子 -> m = 12.4 kg。
        self.m_vehicle = 12.4              # kg，车体 + 四个驱动轮；传感器质量较小，暂不计入
        self.Iz_vehicle = 0.282            # kg*m^2，按车体长方体惯量 + 四轮平移惯量近似得到
        self.R_wheel = 0.07162             # m，轮半径
        self.B_track = 0.271               # m，左右轮距 = 2 * 0.1355
        self.L_wheelbase = 0.240           # m，前后轴距 = 2 * 0.120
        self.h_cg = 0.1195                 # m，base_link 近似离地高度，可用于后续载荷转移/侧翻约束
        self.mu_nominal = 1.0              # Gazebo world 中地面名义摩擦系数 mu/mu2
        self.g_const = 9.81

        # ---------- 沿用版本3已经匹配过的命令响应参数 ----------
        # 注意：因为当前 Gazebo 的 DiffDrive 插件输入仍是 /cmd_vel，而不是轮力矩，
        # 所以 Fx/Mz 采用“命令速度误差 -> 等效力/等效力矩”的灰箱写法。
        self.kv_nominal = 0.999852
        self.kr_nominal = 1.003723
        self.tau_v = 0.061498
        self.tau_r = 0.074421

        # ---------- 地形等效增益，保留版本3模型匹配结果 ----------
        # Gazebo DiffDrive 的速度控制会部分抵消真实重力坡度影响，所以不直接固定为 1.0。
        self.slope_gain = 0.2
        self.cross_slope_gain = 0.0015
        self.tau_y = 3.0
        self.ky_vr = 0.0

        # ---------- 需要后续继续模型匹配/实车辨识的灰箱阻尼参数 ----------
        self.c_rr = 0.03                   # 滚阻系数，包内没有直接给出，先给名义初值
        self.c_d = 0.0                     # 低速小车空气阻力可先忽略
        self.C_y = self.m_vehicle / self.tau_y                   # N/(m/s)，横向滑移阻尼，需要横坡/转弯实验再匹配
        self.C_yr = 0.0                    # N/(rad/s)，r 对横向力的等效耦合，先置 0
        self.k_vxr = 0.0
        self.C_r = 0.05                    # N*m/(rad/s)，横摆阻尼，需要原地转向实验再匹配
        self.C_rv = 0.0                    # N*m/(m/s)，vy 对阻转矩的耦合，先置 0
        

        # 光滑饱和/符号函数参数，避免 IPOPT 面对硬 min/max 或 sign 时不稳定。
        self.sat_eps = 1e-6
        self.smooth_sign_eps = 0.05
        self.smooth_abs_eps = 1e-6

        self.max_slope_rad = 0.45          # 纵坡角限幅，约 ±25.8 deg
        self.max_beta_rad = 0.45           # 横坡角限幅，约 ±25.8 deg
        self.mu_min = 0.1                  # 防止附着参数过小导致除零或不可控
        self.mu_max = 1.5
        self.alpha_filter_window = 5       # 路径高程差分得到的纵坡做滑动平均滤波
        self.beta_filter_window = 5        # 横坡预瞄做滑动平均滤波
        self.use_path_roll_as_beta = True
        self.beta_default = 0.0            # 如果路径姿态没有 roll 信息，则横坡默认为 0

        self.path_ready = False
        self.t_track = None            # 参考时间轴
        self.t_final = 0.0             # 轨迹总时间
        self.start_time = None         # 控制器开始计时时间
        self.t_ref_offset = 0.0        # 初始对齐时的时间偏置

        self.u_prev = np.array([0.0, 0.0])

        self.is_initialized = False

        self.get_logger().info("初始化 CasADi NMPC 求解器：版本4 6状态物理灰箱动力学 + 纵坡 + 横坡...")
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

    def init_casadi_solver(self):
        self.Np = 20
        self.nx = 6       # 版本4：X, Y, theta, vx, vy, r
        self.nu = 2       # v_cmd, omega_cmd
        self.np_terrain = 3  # alpha：路径切向纵坡角；beta：路径法向横坡角；mu：附着系数

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

        alpha = terrain[0]  # 路径切向纵坡角，alpha > 0 表示上坡
        beta = terrain[1]   # 路径法向横坡角
        mu = terrain[2]     # 名义附着系数

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

        # 版本4：物理灰箱动力学模型。
        # 这里仍保持 /cmd_vel 输入，所以不能直接使用轮力矩；
        # 做法是将速度命令误差转换为等效纵向力 Fx，将角速度命令误差转换为等效横摆力矩 Mz。
        X_dot = vx * ca.cos(theta) - vy * ca.sin(theta)
        Y_dot = vx * ca.sin(theta) + vy * ca.cos(theta)
        theta_dot = r

        # 地形总倾角，用于法向载荷近似；mu 也作为预瞄参数传入，默认取仿真地面 mu=1.0。
        mu_eff = ca.fmax(self.mu_min, ca.fmin(self.mu_max, mu))
        gamma = ca.sqrt(alpha**2 + beta**2 + 1e-8)
        normal_factor = ca.cos(gamma)

        # /cmd_vel 到等效纵向力。低速下用 tanh 光滑饱和，避免硬裁剪影响 IPOPT。
        Fx_raw = m_vehicle / tau_v * (kv * v_cmd - vx)
        Fx_max = mu_eff * m_vehicle * g_const * normal_factor
        Fx = Fx_max * ca.tanh(Fx_raw / (Fx_max + sat_eps))

        # 滚阻和空气阻力；滚阻方向与 vx 反向，用光滑 sign 近似。
        smooth_sign_vx = ca.tanh(vx / smooth_sign_eps)
        smooth_abs_vx = ca.sqrt(vx**2 + smooth_abs_eps)
        F_roll = c_rr * m_vehicle * g_const * normal_factor * smooth_sign_vx
        F_drag = c_d * vx * smooth_abs_vx

        # 等效横向轮地力。这里采用阻尼型侧滑力，参数 Cy/Cyr 后续可由横坡或转弯实验辨识。
        Fy = -C_y * vy - C_yr * r

        # /cmd_vel 到等效横摆力矩，并用附着力给出横摆力矩上限。
        Mz_raw = Iz_vehicle / tau_r * (kr * omega_cmd - r)
        Mz_max = mu_eff * m_vehicle * g_const * normal_factor * B_track / 2.0
        Mz = Mz_max * ca.tanh(Mz_raw / (Mz_max + sat_eps))

        # 阻转矩/滑移耗散项。
        M_res = C_r * r + C_rv * vy

        k_vxr = self.k_vxr
        ky_vr = self.ky_vr
        # 刚体动力学：m(vx_dot - vy*r)=Fx-Froll-Fdrag-mg*sin(alpha)
        #             m(vy_dot + vx*r)=Fy-mg*sin(beta)
        #             Iz*r_dot=Mz-Mres
        # slope_gain/cross_slope_gain 保留版本3模型匹配结果，避免 Gazebo 速度插件下坡度补偿过强。
        vx_dot = vy * r + (Fx - F_roll - F_drag) / m_vehicle - slope_gain * g_const * ca.sin(alpha)
        vy_dot = (-k_vxr * vx * r + ky_vr * vx * r + Fy / m_vehicle - cross_slope_gain * g_const * ca.sin(beta))
        r_dot = (Mz - M_res) / Iz_vehicle

        f_continuous = ca.Function(
            'f_continuous',
            [x, u, terrain],
            [ca.vertcat(X_dot, Y_dot, theta_dot, vx_dot, vy_dot, r_dot)]
        )

        # RK4离散化，保持原代码的离散化方式不变。
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
        x0 = ca.SX.sym('x0', self.nx)
        u_prev = ca.SX.sym('u_prev', self.nu)

        J = 0
        g = [X[:, 0] - x0]

        Qdu = ca.DM(np.diag([2.0, 1.0]))

        for k in range(self.Np):
            dx = X[0, k] - X_ref[0, k]
            dy = X[1, k] - X_ref[1, k]
            th_ref = X_ref[2, k]

            # Lead-Lag误差：沿参考切向为 e_t，沿参考法向为 e_n。
            e_t = dx * ca.cos(th_ref) + dy * ca.sin(th_ref)
            e_n = -dx * ca.sin(th_ref) + dy * ca.cos(th_ref)
            e_theta = ca.atan2(ca.sin(X[2, k] - th_ref), ca.cos(X[2, k] - th_ref))

            # 6状态动力学新增：实际纵向速度 vx、横向速度 vy 和横摆角速度 r 的误差。
            vx_err = X[3, k] - X_ref[3, k]
            vy_err = X[4, k] - X_ref[4, k]
            r_err = X[5, k] - X_ref[5, k]

            # 输入命令仍然跟踪参考速度/参考角速度，保证接口与 /cmd_vel 一致。
            v_err = U[0, k] - U_ref[0, k]
            w_err = U[1, k] - U_ref[1, k]

            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])

            J += 40.0 * e_t**2 + 80.0 * e_n**2 + 8.0 * e_theta**2
            J += 8.0 * vx_err**2 + 10.0 * vy_err**2 + 4.0 * r_err**2
            J += 1.0 * v_err**2 + 0.5 * w_err**2
            J += ca.mtimes([du.T, Qdu, du])

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
        J += 15.0 * vx_errN**2 + 20.0 * vy_errN**2 + 8.0 * r_errN**2

        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = ca.vertcat(
            ca.reshape(X_ref, -1, 1),
            ca.reshape(U_ref, -1, 1),
            ca.reshape(Terrain_ref, -1, 1),
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

        # 给动力学状态 vx/vy/r 设置合理边界，避免优化器利用不现实状态降低位置误差。
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

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return

        self.get_logger().info("收到带时间戳参考轨迹，直接建立时间插值...")

        x_pts, y_pts, z_pts, theta_pts, roll_pts, t_pts = [], [], [], [], [], []
        t0_sec = None
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z

            q = pose_stamped.pose.orientation
            theta = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            roll = math.atan2(
                2.0 * (q.w * q.x + q.y * q.z),
                1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            )

            sec = pose_stamped.header.stamp.sec + pose_stamped.header.stamp.nanosec * 1e-9
            if t0_sec is None:
                t0_sec = sec
            t_rel = sec - t0_sec

            x_pts.append(x)
            y_pts.append(y)
            z_pts.append(z)
            theta_pts.append(theta)
            roll_pts.append(roll)
            t_pts.append(t_rel)

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        z_pts = np.array(z_pts)
        theta_pts = np.unwrap(np.array(theta_pts))
        roll_pts = np.unwrap(np.array(roll_pts))
        t_pts = np.array(t_pts)

        keep = np.hstack(([True], np.diff(t_pts) > 1e-6))
        x_pts = x_pts[keep]
        y_pts = y_pts[keep]
        z_pts = z_pts[keep]
        theta_pts = theta_pts[keep]
        roll_pts = roll_pts[keep]
        t_pts = t_pts[keep]

        if len(x_pts) < 2:
            self.get_logger().warn("路径有效点不足，忽略本次路径")
            return

        dt = np.diff(t_pts)
        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        dz = np.diff(z_pts)
        dtheta = np.diff(theta_pts)

        # 由路径高程 z 计算路径切向纵坡 alpha。
        # alpha > 0 表示沿路径前进方向 z 增大，即上坡。
        ds = np.sqrt(dx**2 + dy**2)
        alpha_pts = np.zeros_like(t_pts)
        alpha_pts[1:] = np.arctan2(dz, np.maximum(ds, 1e-6))
        alpha_pts[0] = alpha_pts[1]

        alpha_pts = np.clip(alpha_pts, -self.max_slope_rad, self.max_slope_rad)

        # 高程差分会放大噪声，因此对 alpha 做简单滑动平均。
        if self.alpha_filter_window > 1 and len(alpha_pts) >= self.alpha_filter_window:
            kernel = np.ones(self.alpha_filter_window) / self.alpha_filter_window
            alpha_smooth = np.convolve(alpha_pts, kernel, mode='same')
            half = self.alpha_filter_window // 2
            alpha_smooth[:half] = alpha_pts[:half]
            alpha_smooth[-half:] = alpha_pts[-half:]
            alpha_pts = alpha_smooth

        # 由路径姿态 roll 读取/传递横坡 beta。
        # 注意：仅靠中心线路径的 x/y/z 只能计算纵坡 alpha，不能唯一确定横坡 beta。
        # 因此本代码约定：如果路径生成器把横坡写入 pose.orientation 的 roll，则使用 roll 作为 beta；
        # 否则 beta 默认为 0。
        if self.use_path_roll_as_beta:
            beta_pts = np.array(roll_pts, dtype=float)
        else:
            beta_pts = np.full_like(t_pts, self.beta_default, dtype=float)

        beta_pts = np.clip(beta_pts, -self.max_beta_rad, self.max_beta_rad)

        if self.beta_filter_window > 1 and len(beta_pts) >= self.beta_filter_window:
            kernel = np.ones(self.beta_filter_window) / self.beta_filter_window
            beta_smooth = np.convolve(beta_pts, kernel, mode='same')
            half = self.beta_filter_window // 2
            beta_smooth[:half] = beta_pts[:half]
            beta_smooth[-half:] = beta_pts[-half:]
            beta_pts = beta_smooth

        # 当前 Path 消息不携带附着系数，版本4先使用 Gazebo 地面名义 mu=1.0。
        # 后续如果路径生成器/地形识别模块能提供 mu(s)，只需要在这里替换 mu_pts。
        mu_pts = np.full_like(t_pts, self.mu_nominal, dtype=float)
        mu_pts = np.clip(mu_pts, self.mu_min, self.mu_max)

        # 从带时间戳轨迹中差分估计参考 vx 与参考 r。
        v_pts = np.zeros_like(t_pts)
        omega_pts = np.zeros_like(t_pts)

        v_pts[1:] = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        v_pts[0] = v_pts[1]

        omega_pts[1:] = dtheta / np.maximum(dt, 1e-6)
        omega_pts[0] = omega_pts[1]

        self.t_track = t_pts
        self.t_final = float(t_pts[-1])

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

        self.get_logger().info(
            f"地形预瞄已建立: alpha_min={np.min(alpha_pts):.4f} rad, "
            f"alpha_max={np.max(alpha_pts):.4f} rad, "
            f"beta_min={np.min(beta_pts):.4f} rad, "
            f"beta_max={np.max(beta_pts):.4f} rad, "
            f"mu={self.mu_nominal:.2f}"
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

            # 6状态参考：位置、航向、参考纵向速度、参考横向速度、参考横摆角速度。
            # 路径跟踪希望侧滑尽量小，因此 vy_ref 设为 0。
            X_ref_mod[0, i] = x_ref
            X_ref_mod[1, i] = y_ref
            X_ref_mod[2, i] = theta_ref
            X_ref_mod[3, i] = np.clip(v_ref, self.v_min, self.v_max)
            X_ref_mod[4, i] = 0.0
            X_ref_mod[5, i] = np.clip(omega_ref, self.omega_min, self.omega_max)

            if i < self.Np:
                U_ref_mod[0, i] = np.clip(v_ref, self.v_min, self.v_max)
                U_ref_mod[1, i] = np.clip(omega_ref, self.omega_min, self.omega_max)
                Terrain_ref_mod[0, i] = np.clip(alpha_ref, -self.max_slope_rad, self.max_slope_rad)
                Terrain_ref_mod[1, i] = np.clip(beta_ref, -self.max_beta_rad, self.max_beta_rad)
                Terrain_ref_mod[2, i] = np.clip(mu_ref, self.mu_min, self.mu_max)

        theta_for_mpc = self.theta_act
        while theta_for_mpc - theta_ref_k > math.pi:
            theta_for_mpc -= 2 * math.pi
        while theta_for_mpc - theta_ref_k < -math.pi:
            theta_for_mpc += 2 * math.pi

        # 6状态初值：当前位置、航向、里程计纵向速度、里程计横向速度、里程计横摆角速度。
        # 如果你的 /odom 没有真实 lateral velocity，vy_act 通常接近 0；
        # 后续做横坡模型匹配时，建议确认 odom.twist.twist.linear.y 是否可信。
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

        alpha_now = float(self.traj_interp['alpha_of_t'](t_ref_now))
        beta_now = float(self.traj_interp['beta_of_t'](t_ref_now))
        mu_now = float(self.traj_interp['mu_of_t'](t_ref_now))
        self.get_logger().info(
            f"Δp: {delta_p:.3f} m | t_ref: {t_ref_now:.2f}/{self.t_final:.2f} s | "
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

