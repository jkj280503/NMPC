#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
import casadi as ca
import numpy as np
import math
from scipy.interpolate import interp1d

class NMPCLeadLagNode(Node):
    def __init__(self):
        super().__init__('nmpc_lead_lag_node')
        
        # ===================== 1. ROS 2 参数与通信接口 =====================
        self.dt = 0.05
        self.control_rate = 1.0 / self.dt
        
        # 订阅与发布
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/global_path', self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc_path', 10) # 用于在 Rviz 中可视化预测轨迹
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        # 车辆实时状态
        self.x_act = 0.0
        self.y_act = 0.0
        self.theta_act = 0.0
        self.v_act = 0.0
        
        # 轨迹与解耦进度控制变量
        self.path_ready = False
        self.s_track = None       # 轨迹的累积弧长数组
        self.path_interp = None   # 插值函数集合
        self.max_s = 0.0
        
        self.v_nominal = 1.0           # 标称巡航速度
        self.T_desired = None          # 如果你有总任务时间，就填秒数；没有就保持 None
        self.a_lat_max = 1.2           # 横向加速度上限，可按车体调
        
        # 超前滞后控制参数
        self.t_track = None            # 路径对应的参考时间轴
        self.t_final = 0.0             # 轨迹总时间
        self.start_time = None         # 控制启动时刻
        self.t_ref_offset = 0.0        # 初始对齐时的时间偏置  

        self.u_prev = np.array([0.0, 0.0])  # 上一时刻控制，用于平滑项

        self.is_initialized = False # 用于判断是否是刚接收到轨迹的第一步
        
        self.get_logger().info("初始化 CasADi NMPC 求解器...")
        self.init_casadi_solver()
        self.u0_guess = np.zeros(self.nx * (self.Np + 1) + self.nu * self.Np)
        self.get_logger().info("✅ NMPC Lead-Lag 节点已启动！等待接收全局路径...")

    # ===================== 2. NMPC 求解器构建 (完美移植 MATLAB) =====================
    def init_casadi_solver(self):
        self.Np = 20
        self.nx = 3
        self.nu = 2
        c_nominal = 0.3

        self.v_min, self.v_max = -0.5, 2.0
        self.omega_min, self.omega_max = -math.pi/2, math.pi/2

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)

        x_dot = u[0]*ca.cos(x[2]) - c_nominal * u[1]*ca.sin(x[2])
        y_dot = u[0]*ca.sin(x[2]) + c_nominal * u[1]*ca.cos(x[2])
        theta_dot = u[1]
        f_continuous = ca.Function('f_continuous', [x, u], [ca.vertcat(x_dot, y_dot, theta_dot)])

        k1 = f_continuous(x, u)
        k2 = f_continuous(x + self.dt/2 * k1, u)
        k3 = f_continuous(x + self.dt/2 * k2, u)
        k4 = f_continuous(x + self.dt * k3, u)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        F_discrete = ca.Function('F_discrete', [x, u], [x_next])

        X = ca.SX.sym('X', self.nx, self.Np+1)
        U = ca.SX.sym('U', self.nu, self.Np)
        X_ref = ca.SX.sym('X_ref', self.nx, self.Np+1)
        U_ref = ca.SX.sym('U_ref', self.nu, self.Np)
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

            v_err = U[0, k] - U_ref[0, k]
            w_err = U[1, k] - U_ref[1, k]

            du = U[:, k] - (u_prev if k == 0 else U[:, k-1])

            J += 40.0 * e_t**2 + 80.0 * e_n**2 + 8.0 * e_theta**2
            J += 1.0 * v_err**2 + 0.5 * w_err**2
            J += ca.mtimes([du.T, Qdu, du])

            g.append(X[:, k+1] - F_discrete(X[:, k], U[:, k]))

        dxN = X[0, self.Np] - X_ref[0, self.Np]
        dyN = X[1, self.Np] - X_ref[1, self.Np]
        th_refN = X_ref[2, self.Np]

        e_tN = dxN * ca.cos(th_refN) + dyN * ca.sin(th_refN)
        e_nN = -dxN * ca.sin(th_refN) + dyN * ca.cos(th_refN)
        e_thetaN = ca.atan2(ca.sin(X[2, self.Np] - th_refN), ca.cos(X[2, self.Np] - th_refN))

        J += 120.0 * e_tN**2 + 150.0 * e_nN**2 + 12.0 * e_thetaN**2

        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = ca.vertcat(
            ca.reshape(X_ref, -1, 1),
            ca.reshape(U_ref, -1, 1),
            x0,
            u_prev
        )

        nlp = {'x': opt_variables, 'p': opt_params, 'f': J, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbg = np.zeros(self.nx * (self.Np + 1))
        self.ubg = np.zeros(self.nx * (self.Np + 1))

        lbx = np.full(self.nx * (self.Np + 1), -np.inf)
        ubx = np.full(self.nx * (self.Np + 1), np.inf)
        lbu = np.tile([self.v_min, self.omega_min], self.Np)
        ubu = np.tile([self.v_max, self.omega_max], self.Np)

        self.lb_opt = np.concatenate([lbx, lbu])
        self.ub_opt = np.concatenate([ubx, ubu])

    # ===================== 3. ROS 回调函数 =====================
    def odom_callback(self, msg):
        self.x_act = msg.pose.pose.position.x
        self.y_act = msg.pose.pose.position.y
        # 从四元数提取航向角 yaw
        q = msg.pose.pose.orientation
        self.theta_act = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.v_act = msg.twist.twist.linear.x
    
    def build_time_profile(self, ds, kappa_pts):
        k_abs = np.maximum(np.abs(kappa_pts), 1e-4)

        v_lim_yaw = self.omega_max / k_abs
        v_lim_lat = np.sqrt(self.a_lat_max / k_abs)

        v_lim = np.minimum.reduce([
            np.full_like(kappa_pts, self.v_max),
            v_lim_yaw,
            v_lim_lat
        ])
        v_lim = np.clip(v_lim, 0.1, self.v_max)

        if self.T_desired is None:
            return np.minimum(v_lim, np.full_like(kappa_pts, self.v_nominal))

        T_min = np.sum(ds / np.maximum(v_lim[1:], 0.1))
        if self.T_desired < T_min:
            self.get_logger().warn(
                f"期望总时间 {self.T_desired:.2f}s 小于可行最短时间 {T_min:.2f}s，无法严格按时到达。"
            )
            return v_lim

        lo, hi = 0.05, 1.0
        for _ in range(30):
            beta = 0.5 * (lo + hi)
            T_beta = np.sum(ds / np.maximum(beta * v_lim[1:], 0.1))
            if T_beta > self.T_desired:
                lo = beta
            else:
                hi = beta

        return hi * v_lim

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return

        self.get_logger().info("收到新全局路径，进行弧长/时间参数化处理...")

        x_pts, y_pts, theta_pts = [], [], []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            q = pose_stamped.pose.orientation
            theta = math.atan2(
                2.0 * (q.w*q.z + q.x*q.y),
                1.0 - 2.0 * (q.y*q.y + q.z*q.z)
            )
            x_pts.append(x)
            y_pts.append(y)
            theta_pts.append(theta)

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        theta_pts = np.unwrap(np.array(theta_pts))

        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        ds = np.sqrt(dx**2 + dy**2)

        self.s_track = np.insert(np.cumsum(ds), 0, 0.0)
        self.max_s = self.s_track[-1]

        ds_safe = np.where(ds == 0, 1e-6, ds)
        dtheta = np.diff(theta_pts)
        kappa_pts = np.insert(dtheta / ds_safe, 0, 0.0)

        self.path_interp = {
            'x': interp1d(self.s_track, x_pts, bounds_error=False, fill_value="extrapolate"),
            'y': interp1d(self.s_track, y_pts, bounds_error=False, fill_value="extrapolate"),
            'theta': interp1d(self.s_track, theta_pts, bounds_error=False, fill_value="extrapolate"),
            'kappa': interp1d(self.s_track, kappa_pts, bounds_error=False, fill_value="extrapolate"),
        }

        v_profile = self.build_time_profile(ds, kappa_pts)

        dt_seg = ds / np.maximum(v_profile[1:], 0.1)
        self.t_track = np.insert(np.cumsum(dt_seg), 0, 0.0)
        self.t_final = float(self.t_track[-1])

        omega_profile = v_profile * kappa_pts

        self.path_interp['s_of_t'] = interp1d(
            self.t_track, self.s_track,
            bounds_error=False,
            fill_value=(0.0, self.max_s)
        )
        self.path_interp['t_of_s'] = interp1d(
            self.s_track, self.t_track,
            bounds_error=False,
            fill_value=(0.0, self.t_final)
        )
        self.path_interp['v_of_t'] = interp1d(
            self.t_track, v_profile,
            bounds_error=False,
            fill_value=(float(v_profile[0]), float(v_profile[-1]))
        )
        self.path_interp['omega_of_t'] = interp1d(
            self.t_track, omega_profile,
            bounds_error=False,
            fill_value=(float(omega_profile[0]), float(omega_profile[-1]))
        )

        self.path_ready = True
        self.is_initialized = False

    # ===================== 4. 主控制循环 =====================
    def control_loop(self):
        if not self.path_ready:
            return

        if not self.is_initialized:
            dists = np.sqrt(
                (self.path_interp['x'](self.s_track) - self.x_act)**2 +
                (self.path_interp['y'](self.s_track) - self.y_act)**2
            )
            s0 = float(self.s_track[np.argmin(dists)])
            self.t_ref_offset = float(self.path_interp['t_of_s'](s0))
            self.start_time = self.get_clock().now()
            self.u_prev = np.array([self.v_act, 0.0])
            self.is_initialized = True
            self.get_logger().info(f"初始对齐完毕，s0 = {s0:.2f} m, t0 = {self.t_ref_offset:.2f} s")
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        t_ref_now = float(np.clip(self.t_ref_offset + elapsed, 0.0, self.t_final))

        s_curr = float(self.path_interp['s_of_t'](t_ref_now))
        x_ref_k = float(self.path_interp['x'](s_curr))
        y_ref_k = float(self.path_interp['y'](s_curr))
        theta_ref_k = float(self.path_interp['theta'](s_curr))

        dx = self.x_act - x_ref_k
        dy = self.y_act - y_ref_k
        delta_p = dx * math.cos(theta_ref_k) + dy * math.sin(theta_ref_k)

        goal_x = float(self.path_interp['x'](self.max_s))
        goal_y = float(self.path_interp['y'](self.max_s))
        goal_dist = math.hypot(self.x_act - goal_x, self.y_act - goal_y)

        if t_ref_now >= self.t_final and goal_dist < 0.10:
            self.stop_robot()
            self.get_logger().info("按计划到达终点！", once=True)
            return

        X_ref_mod = np.zeros((self.nx, self.Np + 1))
        U_ref_mod = np.zeros((self.nu, self.Np))

        for i in range(self.Np + 1):
            t_i = min(t_ref_now + i * self.dt, self.t_final)
            s_i = float(self.path_interp['s_of_t'](t_i))

            X_ref_mod[0, i] = float(self.path_interp['x'](s_i))
            X_ref_mod[1, i] = float(self.path_interp['y'](s_i))
            X_ref_mod[2, i] = float(self.path_interp['theta'](s_i))

            if i < self.Np:
                U_ref_mod[0, i] = float(self.path_interp['v_of_t'](t_i))
                U_ref_mod[1, i] = float(self.path_interp['omega_of_t'](t_i))

        theta_for_mpc = self.theta_act
        while theta_for_mpc - theta_ref_k > math.pi:
            theta_for_mpc -= 2 * math.pi
        while theta_for_mpc - theta_ref_k < -math.pi:
            theta_for_mpc += 2 * math.pi

        opt_params_k = np.concatenate([
            X_ref_mod.flatten(order='F'),
            U_ref_mod.flatten(order='F'),
            np.array([self.x_act, self.y_act, theta_for_mpc]),
            self.u_prev
        ])

        res = self.solver(
            x0=self.u0_guess,
            p=opt_params_k,
            lbg=self.lbg,
            ubg=self.ubg,
            lbx=self.lb_opt,
            ubx=self.ub_opt
        )

        opt_sol = np.array(res['x']).flatten()
        self.u0_guess = opt_sol

        u_opt = opt_sol[self.nx * (self.Np + 1):].reshape(self.nu, self.Np, order='F')
        v_cmd = float(u_opt[0, 0])
        omega_cmd = float(u_opt[1, 0])

        self.u_prev = np.array([v_cmd, omega_cmd])

        cmd_msg = Twist()
        cmd_msg.linear.x = v_cmd
        cmd_msg.angular.z = omega_cmd
        self.cmd_pub.publish(cmd_msg)

        self.get_logger().info(
            f"Δp: {delta_p:.3f} m | t_ref: {t_ref_now:.2f}/{self.t_final:.2f} s | "
            f"cmd_v: {v_cmd:.2f} | cmd_w: {omega_cmd:.2f}"
        )

        self.publish_mpc_trajectory(opt_sol)

    def publish_mpc_trajectory(self, opt_sol):
        # 提取预测的空间轨迹并发布给 Rviz
        X_opt = opt_sol[:self.nx * (self.Np + 1)].reshape(self.nx, self.Np + 1, order='F')
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom" # 根据你的TF树调整
        
        for i in range(self.Np + 1):
            pose = PoseStamped()
            pose.pose.position.x = float(X_opt[0, i])
            pose.pose.position.y = float(X_opt[1, i])
            path_msg.poses.append(pose)
            
        self.mpc_path_pub.publish(path_msg)

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