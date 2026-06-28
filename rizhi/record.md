# 创新点
针对小型实验车在带时间戳轨迹跟踪中存在纵向滞后、检查点到达时间误差大的问题，构建了考虑纵向响应滞后的动力学 NMPC 控制器，并引入基于纵向投影误差的时间补偿机制，实现了对参考速度序列的在线修正。
为评价控制器对上层多车时空规划的支撑能力，设计了检查点准时通过实验，采用检查点到达时间误差、终点到达时间误差和纵横向跟踪误差作为评价指标。
通过实车开环实验辨识纵向速度响应参数，提高了预测模型与实验车实际响应的一致性。


# 调试记录
## matlab
### 文件与结果
原始文件为mubiaoguiji1.m和qiujiegengxin0.m
测试结果为"测试结果0.png"
### 测试分析
1.阈值设定太大，无法触发超前滞后补偿机制；
2.x,y跟踪误差图有误，结果变成了图块。
### 参数修改
对qiujiegengxin.m进行修改：
#对272，279行代码进行修改
#plot(t, x_history(1,:)' - Ref_Trajectory(:,1), 'b-', 'LineWidth',1.2); % 加了转置号 '
#plot(t, x_history(2,:)' - Ref_Trajectory(:,2), 'b-', 'LineWidth',1.2); % 加了转置号 '

#对106-107行代码进行修改
delta_thresh = 0.05;  % 缩小到 5cm 触发补偿
kp = 1.0;             % 增大补偿力度，让速度变化更明显

% 第 17 行，先统一滑移系数
c_nominal = 0.3; 

% 第 21 行，增大对角速度的惩罚，抑制震荡
R = diag([0.5, 0.5]); 

% 第 111 行，先关闭噪声，验证纯算法逻辑
enable_noise = false;

### 文件与结果
修改后文件为qiujiegengxin1.m
测试结果为"测试结果1.png"

### 测试分析
车辆在直线段的表现还行，但在坐标(3, 2.5)附近的第一个急弯处，蓝线（实际轨迹）出现了极其严重的外扩，随后花了好几秒钟才勉强切回红线。
速差滑移车在急弯处本来就容易侧滑（由于存在滑移系数 $c=0.3$），此时 NMPC 接收到了“加速”的指令，导致车辆以过高的速度冲入弯道，离心力和滑移加剧，直接导致了子图1中严重的向外甩尾！
【在高速公路（微小曲率）上，落后了踩油门是对的；但在速差机器人的急弯中，落后了踩油门是致命的！】 

### 参数修改
思路：1.修改 Lead-Lag 速度补偿逻辑：当车辆在急弯处发生滞后时，我们不应该让它加速去送死，而是应该降低参考速度，让 NMPC 把宝贵的控制力矩全部分配给转向（角速度 $\omega$），等车头顺过来之后再走。
% ========== 2. 改进型速度补偿计算 (适配滑移转向) ==========
    v_comp = 0;
    if delta_p > delta_thresh
        % 车辆超前：适当加速参考点（或让车辆减速等待），这里输出正补偿来拉高参考速度
        v_comp = kp * (delta_p - delta_thresh); 
    elseif delta_p < -delta_thresh
        % 车辆滞后：说明过弯困难，必须【降低】参考速度，等待车辆转向上来
        v_comp = kp * (delta_p + delta_thresh); % 注意这里算出来是负数！
    end
    % 限制补偿幅度，防止失控
    v_comp = max(min(v_comp, 0.5), -0.5); 
    v_comp_history(k) = v_comp;

2.调高 NMPC 的空间跟踪权重：你目前的权重是 Q = diag([10, 10, 1])。NMPC 对纵向和横向误差的惩罚一视同仁。引入 Lead-Lag 后，我们在时间（纵向）上已经有了外环补偿，所以内环的 NMPC 应该死死咬住空间坐标。
% 权重矩阵：极大地增加对位置误差的惩罚，迫使求解器优先贴合轨迹
Q = diag([50, 50, 5]);


### 文件与结果
修改后文件为qiujiegengxin2.m
测试结果为"测试结果2.png"

### 测试分析

## ros2
## 第一版文件
测试轨迹：send_test_path1.py
时空轨迹控制算法：nmpc_lead_lag_node1.py
车辆/空间模型：lino_ws文件夹

启动命令：
环境终端
source ~/nmpc/lino_ws/install/setup.bash
export LINOROBOT2_BASE=4wd
ros2 launch linorobot2_gazebo gazebo.launch.py

ros2 launch linorobot2_gazebo gazebo.launch.py spawn_z:=0.1

python3 bag_to_csv.py \
  --bag ~/rizhi/rosbag2_2026_04_28-21_00_34 \
  --out gazebo_openloop_v1.csv

nmpc终端：
source /opt/ros/jazzy/setup.bash
cd ~/nmpc/nmpc_ws
python3 src/nmpc_controller/nmpc_controller/nmpc_lead_lag_node0.py

轨迹终端：
cd ~/nmpc/nmpc_ws
source install/setup.bash
python3 src/nmpc_controller/nmpc_controller/send_test_path.py 

rviz2终端：
rviz2 -d ~/experiments/rviz_nmpc_config.rviz

plotjuggler终端：
ros2 run plotjuggler plotjuggler

修改：
1.目标轨迹补全时间戳，规范声明的qos策略//
2.修正nmpc的代码缺陷

## 第二版文件
测试轨迹：send_test_path1.py
时空轨迹控制算法：nmpclead_lag_node2.py

## 修改
考虑到目前的思路虽然在算法上可行，但结果并不好，此外更严重的问题是目前的思路是做两层控制，由超前滞后机制对目标车辆进行控制，而nmpc求解器对实际车辆进行控制，解决外部补偿和预测控制的冲突问题，但是这样跟踪的却是修改后的目标轨迹，偏离了初始方向！！！！！

换一种思路，不要把机制作用在参考轨迹上，还是得去修改实际小车，但不再让这个机制单独作为一部分，把他放到nmpc里面去，作为里面的改写的一部分，修改nmpc的代价函数，把时间考虑进来。

## 第三版文件
测试轨迹：send_test_path1.py
时空轨迹控制算法：nmpclead_lag_node3.py

### 修改
对代码进行优化修改，使结果更清晰

## 第四版文件
测试轨迹：send_test_path1.py,send_test_path3.py
时空轨迹控制算法：nmpclead_lag_node2.py

### 修改
收到参考轨迹，在第一次控制前进行时间对齐，参考时间按照真实时间推进，比较实际小车状态和目标状态，进行nmpc优化控制


通过直线和正弦曲线轨迹的仿真，并将机制关闭和开启两种情况下的结果分析，横向的跟踪效果差距不大，但纵向上效果还是有的。
关闭机制进行曲线轨迹时，时间误差为0.832s,开启时为0.031s,并且开启机制时e_t曲线更接近0,滞后现象更小了，差距还是比较明显的。
但还不能说明他是普遍有效的，看看加上不同初始偏差，不同扰动之后的区别如何。

## 第四版文件
调整机制参数，将运动学模型修改为动力学模型，先改为6阶状态灰箱模型看看，由原本的[x,y,sita]转变为[x,y,ψ, u, v, r]
文件：nmpc_donglixue_node1.py

### 问题
修改后的版本出现在运行时顿挫感明显比运动学模型版本的更强，到达误差也比修改前的更大，可能是修改后的模型与仿真车辆模型没有贴合，
修改命令参考权重;让v_body_act为0,削弱odom.linear.y的干扰；增大控制增量惩罚Qdu,辨识数据修改tau_u/tau_r


### 第五版文件
nmpc_donglixue_node2.py

顿挫感更小了，不过到达误差还是比运动学模型的时候大，先显式引入坡度因素看看，把坡度成功引入之后再去调整。

将纵坡引入u_dot,侧坡引入v_dot,先只做常值坡度的拓展实现，不引入坡度预瞄，后续看效果进行修改


python3 ~/nmpc/nmpc_exp/contrast/nmpc_csv_recorder_fixed.py  --name dynamic_ll_01  --out ~/nmpc/nmpc_exp/contrast/tuboshu   --t_final 12.118867   --goal_tolerance 0.10   --auto_stop

source /opt/ros/jazzy/setup.bash
python3 ~/nmpc/nmpc_exp/loop_cmd/new/open_loop_cmd_publisher.py --case A2


ros2 bag record \
  -o ~/nmpc/nmpc_exp/loop_cmd/tuboshu/rosbag/A1 \
  /odom \                         
  /cmd_vel \                         
  /joint_states \
  /clock \
  /tf \
  /tf_static \
  /openloop_debug/alpha \
  /openloop_debug/beta \
  /openloop_debug/mu \
  /openloop_debug/case \
  /openloop_debug/phase


source ~/nmpc_venv/bin/activate
mkdir -p ~/nmpc/nmpc_exp/loop_cmd/tuboshu/csv_simtime

for bag in ~/nmpc/nmpc_exp/loop_cmd/tuboshu/rosbag/*; do
  if [ -f "$bag/metadata.yaml" ]; then
    name=$(basename "$bag")
    echo "Converting $name with Gazebo sim time"

    python3 ~/nmpc/nmpc_exp/loop_cmd/tuboshu/rosbag_to_csv_openloop_simtime.py \
                    
      --bag "$bag" \                                          
      --out ~/nmpc/nmpc_exp/loop_cmd/tuboshu/csv_simtime/"$name" \
      --time-source clock
  fi
done
