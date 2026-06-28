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
        
        self.v_const = 1.0        # 标称速度 (m/s)
        self.s_ref_actual = 0.0   # 虚拟目标点当前推进的弧长
        self.v_ref_actual = 1.0   # 虚拟目标点当前真实速度
        
        # 超前滞后控制参数
        self.delta_thresh = 0.05
        self.T_param = 1.0
        self.is_initialized = False # 用于判断是否是刚接收到轨迹的第一步
        
        self.get_logger().info("初始化 CasADi NMPC 求解器...")
        self.init_casadi_solver()
        self.get_logger().info("✅ NMPC Lead-Lag 节点已启动！等待接收全局路径...")

    # ===================== 2. NMPC 求解器构建 (完美移植 MATLAB) =====================
    def init_casadi_solver(self):
        self.Np = 20
        self.nx = 3
        self.nu = 2
        c_nominal = 0.3
        
        # 权重与约束
        Q = np.diag([10.0, 10.0, 1.0])
        R = np.diag([0.5, 0.5])
        self.v_min, self.v_max = -0.5, 2.0
        self.omega_min, self.omega_max = -math.pi/2, math.pi/2
        
        # 符号变量
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        
        # 运动学方程 (速差滑移)
        x_dot = u[0]*ca.cos(x[2]) - c_nominal * u[1]*ca.sin(x[2])
        y_dot = u[0]*ca.sin(x[2]) + c_nominal * u[1]*ca.cos(x[2])
        theta_dot = u[1]
        f_continuous = ca.Function('f_continuous', [x, u], [ca.vertcat(x_dot, y_dot, theta_dot)])
        
        # RK4 离散化
        k1 = f_continuous(x, u)
        k2 = f_continuous(x + self.dt/2 * k1, u)
        k3 = f_continuous(x + self.dt/2 * k2, u)
        k4 = f_continuous(x + self.dt * k3, u)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        F_discrete = ca.Function('F_discrete', [x, u], [x_next])
        
        # 优化变量与参数
        X = ca.SX.sym('X', self.nx, self.Np+1)
        U = ca.SX.sym('U', self.nu, self.Np)
        X_ref = ca.SX.sym('X_ref', self.nx, self.Np+1)
        U_ref = ca.SX.sym('U_ref', self.nu, self.Np)
        x0 = ca.SX.sym('x0', self.nx)
        
        J = 0
        g = [X[:, 0] - x0]
        
        for k in range(self.Np):
            state_err = X[:, k] - X_ref[:, k]
            J += ca.mtimes([state_err.T, Q, state_err])
            ctrl_err = U[:, k] - U_ref[:, k]
            J += ca.mtimes([ctrl_err.T, R, ctrl_err])
            g.append(X[:, k+1] - F_discrete(X[:, k], U[:, k]))
            
        opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opt_params = ca.vertcat(ca.reshape(X_ref, -1, 1), ca.reshape(U_ref, -1, 1), x0)
        
        nlp = {'x': opt_variables, 'p': opt_params, 'f': J, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 边界设置
        self.lbg = np.zeros(self.nx * (self.Np + 1))
        self.ubg = np.zeros(self.nx * (self.Np + 1))
        
        lbx = np.full(self.nx * (self.Np + 1), -ca.inf)
        ubx = np.full(self.nx * (self.Np + 1), ca.inf)
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

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return
        
        self.get_logger().info("收到新全局路径，进行弧长参数化处理...")
        # 提取坐标并计算累积弧长 s
        x_pts, y_pts, theta_pts = [], [], []
        for i, pose_stamped in enumerate(msg.poses):
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            q = pose_stamped.pose.orientation
            theta = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            x_pts.append(x)
            y_pts.append(y)
            theta_pts.append(theta)
            
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        theta_pts = np.unwrap(np.array(theta_pts)) # 防止角度跳变
        
        # 计算 s 数组
        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        ds = np.sqrt(dx**2 + dy**2)
        self.s_track = np.insert(np.cumsum(ds), 0, 0.0)
        self.max_s = self.s_track[-1]
        
        # 计算曲率 kappa (dtheta / ds)
        dtheta = np.diff(theta_pts)
        # 处理避免 ds 为 0 的除法
        ds_safe = np.where(ds == 0, 1e-6, ds)
        kappa_pts = np.insert(dtheta / ds_safe, 0, 0.0) 
        
        # 构建插值函数 (将离散点变为空域连续函数)
        self.path_interp = {
            'x': interp1d(self.s_track, x_pts, fill_value="extrapolate"),
            'y': interp1d(self.s_track, y_pts, fill_value="extrapolate"),
            'theta': interp1d(self.s_track, theta_pts, fill_value="extrapolate"),
            'kappa': interp1d(self.s_track, kappa_pts, fill_value="extrapolate")
        }
        
        self.path_ready = True
        self.is_initialized = False # 触发重新寻找起点的逻辑

    # ===================== 4. 主控制循环 =====================
    def control_loop(self):
        if not self.path_ready:
            return
        
        # 如果是刚接手路径，先通过距离寻找离车最近的 s，对齐进度条
        if not self.is_initialized:
            dists = np.sqrt((self.path_interp['x'](self.s_track) - self.x_act)**2 + 
                            (self.path_interp['y'](self.s_track) - self.y_act)**2)
            self.s_ref_actual = self.s_track[np.argmin(dists)]
            self.v_ref_actual = self.v_const
            self.is_initialized = True
            self.get_logger().info(f"起点对齐完毕，初始 s = {self.s_ref_actual:.2f} m")

        # 终点判定
        if self.s_ref_actual >= self.max_s - 0.1:
            self.stop_robot()
            self.get_logger().info("到达终点！", once=True)
            return

        # ========== 1. 基于弧长提取解耦后的目标点 ==========
        s_curr = np.clip(self.s_ref_actual, 0, self.max_s)
        x_ref_k = float(self.path_interp['x'](s_curr))
        y_ref_k = float(self.path_interp['y'](s_curr))
        theta_ref_k = float(self.path_interp['theta'](s_curr))
        
        # ========== 2. 计算纵向误差 delta_p ==========
        dx = self.x_act - x_ref_k
        dy = self.y_act - y_ref_k
        u_tau_x = math.cos(theta_ref_k)
        u_tau_y = math.sin(theta_ref_k)
        delta_p = dx * u_tau_x + dy * u_tau_y
        
        # ========== 3. Lead-Lag 加速度补偿 ==========
        a_comp = 0.0
        if delta_p > self.delta_thresh:    # 车超前
            a_comp = 2 * (delta_p - self.delta_thresh) / (self.T_param**2)
        elif delta_p < -self.delta_thresh: # 车滞后
            a_comp = 2 * (delta_p + self.delta_thresh) / (self.T_param**2)
            
        a_comp = np.clip(a_comp, -1.0, 1.0)
        
        # 动力学重构
        a_ref_total = a_comp + 1.5 * (self.v_const - self.v_ref_actual)
        self.v_ref_actual += a_ref_total * self.dt
        self.v_ref_actual = np.clip(self.v_ref_actual, 0.1, self.v_max)
        self.s_ref_actual += self.v_ref_actual * self.dt
        
        # ========== 4. 构建 Np 步时空对齐序列 ==========
        X_ref_mod = np.zeros((self.nx, self.Np + 1))
        U_ref_mod = np.zeros((self.nu, self.Np))
        
        s_future = self.s_ref_actual
        for i in range(self.Np + 1):
            s_future_clip = np.clip(s_future, 0, self.max_s)
            X_ref_mod[0, i] = float(self.path_interp['x'](s_future_clip))
            X_ref_mod[1, i] = float(self.path_interp['y'](s_future_clip))
            X_ref_mod[2, i] = float(self.path_interp['theta'](s_future_clip))
            
            if i < self.Np:
                kappa = float(self.path_interp['kappa'](s_future_clip))
                # 核心：过弯角速度必须随实际速度缩放
                omega_scaled = self.v_ref_actual * kappa
                U_ref_mod[0, i] = self.v_ref_actual
                U_ref_mod[1, i] = omega_scaled
                
            s_future += self.v_ref_actual * self.dt
            
        # ========== 5. 调用 CasADi 求解 ==========
        opt_params_k = np.concatenate([
            X_ref_mod.flatten(),
            U_ref_mod.flatten(),
            np.array([self.x_act, self.y_act, self.theta_act])
        ])
        
        res = self.solver(x0=np.zeros(self.nx*(self.Np+1) + self.nu*self.Np), 
                          p=opt_params_k, lbg=self.lbg, ubg=self.ubg, 
                          lbx=self.lb_opt, ubx=self.ub_opt)
        
        opt_sol = np.array(res['x']).flatten()
        
        # 提取第一个控制指令
        u_opt = opt_sol[self.nx * (self.Np + 1):].reshape(self.nu, self.Np, order='F')
        v_cmd = float(u_opt[0, 0])
        omega_cmd = float(u_opt[1, 0])
        
        # 发布指令
        cmd_msg = Twist()
        cmd_msg.linear.x = v_cmd
        cmd_msg.angular.z = omega_cmd
        self.cmd_pub.publish(cmd_msg)
        
        # 调试打印
        self.get_logger().info(f"Δp: {delta_p: .3f}m | a_comp: {a_comp: .2f} | "
                               f"v_ref: {self.v_ref_actual:.2f} | cmd_v: {v_cmd:.2f} | cmd_w: {omega_cmd:.2f}")
                               
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