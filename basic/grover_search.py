# 导入 Qiskit 和其他必要的库
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 定义问题 ---
# 定义我们要搜索的量子比特数量
n = 10
# 定义我们要在 2^n 个状态中搜索的目标状态 (一个10位的二进制字符串)
target_state = '1011011001'

# --- 2. 构建 Grover 算法的组件 ---

# --- 2a. 构建预言机 (Oracle) ---
# 预言机的作用是“标记”出我们正在寻找的目标状态。
# 它通过给目标状态施加一个负的相位 (-1) 来实现这一点，而其他所有状态的相位保持不变。
# 我们通过一个多控 Z 门 (multi-controlled Z gate) 来实现。
oracle = QuantumCircuit(n, name='Oracle')

# 要构建一个可以识别 '1011011001' 的多控Z门，我们首先需要
# 将那些在 target_state 中为 '0' 的量子比特用 X 门翻转过来。
# 这样，只有当输入是 '1011011001' 时，所有量子比特才会都变成 |1> 状态。
zero_indices = [i for i, bit in enumerate(target_state) if bit == '0']
oracle.x(zero_indices)

# 现在，我们施加一个由所有 n-1 个量子比特控制的 Z 门到最后一个量子比特上。
# 这可以通过一个多控 X 门 (mcx) 和两个 H 门夹住目标量子比特来实现。
oracle.h(n-1)
oracle.mcx(list(range(n-1)), n-1) # 多控 Toffoli (X) 门
oracle.h(n-1)

# 最后，我们再次用 X 门将之前翻转的量子比特恢复原状。
oracle.x(zero_indices)


# --- 2b. 构建扩散器 (Diffuser / Amplification) ---
# 扩散器的作用是放大被预言机标记出来的目标状态的振幅。
# 它对所有状态的平均振幅进行反射操作。
diffuser = QuantumCircuit(n, name='Diffuser')

# 扩散器的标准构建步骤：
# 1. 对所有量子比特施加 H 门
diffuser.h(range(n))
# 2. 对所有量子比特施加 X 门
diffuser.x(range(n))
# 3. 施加一个多控 Z 门 (和预言机中的类似)
diffuser.h(n-1)
diffuser.mcx(list(range(n-1)), n-1)
diffuser.h(n-1)
# 4. 再次对所有量子比特施加 X 门
diffuser.x(range(n))
# 5. 再次对所有量子比特施加 H 门
diffuser.h(range(n))


# --- 3. 构建完整的 Grover 电路 ---

# 对于 N = 2^n 个状态，最佳的迭代次数约为 (π/4) * sqrt(N)
iterations = int(np.pi / 4 * np.sqrt(2**n))
print(f"量子比特数量: {n}, 状态总数: {2**n}, 最佳迭代次数: {iterations}")

# 创建一个包含 n 个量子比特和 n 个经典比特的量子电路
grover_circuit = QuantumCircuit(n, n)

# 步骤 1: 初始化。对所有量子比特施加 H 门，创造一个均匀的叠加态。
grover_circuit.h(range(n))

# 步骤 2: 重复应用预言机和扩散器
for _ in range(iterations):
    grover_circuit.append(oracle, range(n))
    grover_circuit.append(diffuser, range(n))

# 步骤 3: 测量。将量子比特的最终状态测量并存储到经典比特中。
grover_circuit.measure(range(n), range(n))

# --- 4. 可视化与模拟 ---

# 将最终的电路图绘制并保存为图片文件
# 注意：对于10个量子比特和25次迭代，电路图会非常大且难以阅读
print("正在生成电路图 (这可能需要一些时间)...")
grover_circuit.draw('mpl', filename='./basic/grover_search_10qubit.png', fold=-1)
print("电路图已保存到 grover_search_10qubit.png")

# 使用 Qiskit Aer 模拟器来运行我们的电路
print("开始量子电路模拟...")
simulator = Aer.get_backend('qasm_simulator')
# 将电路转换为模拟器可以理解的格式
compiled_circuit = transpile(grover_circuit, simulator)
# 运行模拟，'shots' 表示我们重复这个实验多少次
result = simulator.run(compiled_circuit, shots=2048).result()
# 获取测量结果的统计
counts = result.get_counts()

# 将结果的直方图绘制并保存为图片
plot_histogram(counts)
plt.savefig('./basic/grover_results_10qubit.png')

print(f"模拟完成！最可能的结果应该是目标状态: {target_state}")
print("结果统计直方图已保存到 grover_results_10qubit.png")