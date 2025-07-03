# 导入 Qiskit 和其他必要的库
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --- CNOT 门与贝尔态纠缠演示 ---

# 1. CNOT 门简介
# CNOT 门，全称为“受控非门”(Controlled-NOT)，是量子计算中最基础和最重要的双量子比特门之一。
# 它有两个输入：一个“控制”量子比特 (control qubit) 和一个“目标”量子比特 (target qubit)。
#
# 工作原理:
# - 如果控制量子比特是 |0⟩，那么目标量子比特保持不变。
# - 如果控制量子比特是 |1⟩，那么目标量子比特的状态会翻转 (X 门操作，即 |0⟩ 变为 |1⟩，|1⟩ 变为 |0⟩)。
#
# 这个简单的规则使得 CNOT 门能够创造出一种奇特的量子现象——量子纠缠。

# 2. 创建贝尔态 (Bell State)
# 贝尔态是最简单的量子纠缠态。当两个量子比特处于贝尔态时，对其中一个的测量结果会瞬间影响到另一个，
# 无论它们相距多远。这被称为“幽灵般的超距作用”。
#
# 我们将创建最著名的一种贝尔态 |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
# 这意味着测量结果有 50% 的概率是 '00'，50% 的概率是 '11'，绝不会出现 '01' 或 '10'。
# 这种状态的关联性就是纠缠的体现。

# --- 3. 构建量子电路 ---

# 创建一个包含 2 个量子比特和 2 个经典比特的量子电路
# q[0] 是控制位, q[1] 是目标位
qc = QuantumCircuit(2, 2, name="BellState")

# 步骤 a: 将第一个量子比特 (q[0]) 置于叠加态
# 我们使用哈达玛门 (Hadamard Gate) 来实现。
# 初始状态: |00⟩
# H 门作用于 q[0] 后，q[0] 变为 (|0⟩ + |1⟩)/sqrt(2)。
# 整个系统的状态变为: (|00⟩ + |10⟩) / sqrt(2)
print("步骤 1: 对控制位 q[0] 应用哈达玛门 (H)，使其进入叠加态。")
qc.h(0)

# 步骤 b: 应用 CNOT 门
# 我们将 q[0] 作为控制位，q[1] 作为目标位。
# - 当 q[0] 是 |0⟩ 时 (50% 概率), q[1] 不变，仍为 |0⟩。系统部分状态为 |00⟩。
# - 当 q[0] 是 |1⟩ 时 (50% 概率), q[1] 翻转，从 |0⟩ 变为 |1⟩。系统部分状态为 |11⟩。
#
# 叠加起来，系统的最终状态就是 (|00⟩ + |11⟩) / sqrt(2)，即贝尔态。
print("步骤 2: 应用 CNOT 门，q[0] 为控制位，q[1] 为目标位，创造纠缠。")
qc.cx(0, 1) # cx 是 CNOT 门的指令

# 步骤 c: 测量
# 我们测量两个量子比特，并将结果存储在经典比特中。
print("步骤 3: 测量两个量子比特。")
qc.measure([0, 1], [0, 1])

# --- 4. 可视化与模拟 ---

# 绘制电路图
print("\n正在生成电路图...")
circuit_drawing = qc.draw('mpl')
circuit_drawing.savefig('./basic/cnot_gate_circuit.png')
print("电路图已保存到 ./basic/cnot_gate_circuit.png")

# 使用 Qiskit Aer 模拟器运行电路
print("\n开始量子电路模拟...")
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit, shots=1024).result()
counts = result.get_counts()

# 打印和绘制结果
print("\n模拟结果统计:")
print(counts)
print("\n从结果可以看出，我们只得到了 '00' 和 '11' 两种结果，且概率各占约 50%。")
print("这证明了两个量子比特已经成功纠缠在了一起。")

# 绘制直方图
plot_histogram(counts)
plt.savefig('./basic/cnot_gate_results.png')
print("\n结果统计直方图已保存到 ./basic/cnot_gate_results.png")