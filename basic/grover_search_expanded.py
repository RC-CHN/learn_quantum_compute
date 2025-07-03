# 导入 Qiskit 和其他必要的库
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def run_grover_search(target_state, iterations=None):
    """
    执行 Grover 搜索算法，并生成展开的电路图。

    参数:
    target_state (str): 要搜索的目标二进制字符串。
    iterations (int, optional): Grover 迭代的次数。如果为 None，则自动计算最佳次数。
    """
    n = len(target_state)
    
    # 预言机和扩散器将直接构建在主电路中

    # --- 计算迭代次数 ---
    if iterations is None:
        iterations = int(np.pi / 4 * np.sqrt(2**n))
    
    print(f"量子比特数量: {n}")
    print(f"目标状态: '{target_state}'")
    print(f"状态总数: {2**n}")
    print(f"Grover 迭代次数: {iterations}")

    # --- 构建完整电路 ---
    grover_circuit = QuantumCircuit(n, n)
    grover_circuit.h(range(n))

    # --- 构建展开的 Grover 迭代 ---
    zero_indices = [i for i, bit in enumerate(target_state) if bit == '0']
    for i in range(iterations):
        grover_circuit.barrier()
        print(f"构建迭代 {i+1}/{iterations}...")
        # --- 预言机 (Oracle) ---
        if zero_indices:
            grover_circuit.x(zero_indices)
        grover_circuit.h(n-1)
        grover_circuit.mcx(list(range(n-1)), n-1)
        grover_circuit.h(n-1)
        if zero_indices:
            grover_circuit.x(zero_indices)

        grover_circuit.barrier()

        # --- 扩散器 (Diffuser) ---
        grover_circuit.h(range(n))
        grover_circuit.x(range(n))
        grover_circuit.h(n-1)
        grover_circuit.mcx(list(range(n-1)), n-1)
        grover_circuit.h(n-1)
        grover_circuit.x(range(n))
        grover_circuit.h(range(n))

    grover_circuit.measure(range(n), range(n))

    # --- 可视化与模拟 ---
    print("正在生成展开的电路图 (这可能需要一些时间)...")
    output_filename_circuit = f'./basic/grover_search_expanded_{n}qubit.png'
    try:
        grover_circuit.draw('mpl', filename=output_filename_circuit, fold=-1)
        print(f"电路图已保存到 {output_filename_circuit}")
    except Exception as e:
        print(f"无法生成电路图: {e}")


    print("开始量子电路模拟...")
    simulator = Aer.get_backend('qasm_simulator')
    compiled_circuit = transpile(grover_circuit, simulator)
    result = simulator.run(compiled_circuit, shots=100).result()
    counts = result.get_counts()

    output_filename_results = f'./basic/grover_results_expanded_{n}qubit.png'
    plot_histogram(counts)
    plt.savefig(output_filename_results)

    print(f"模拟完成！最可能的结果应该是目标状态: {target_state}")
    print(f"结果统计直方图已保存到 {output_filename_results}")


def main():
    parser = argparse.ArgumentParser(description="使用 Grover 算法在量子计算机上进行搜索，并生成展开的电路图。")
    parser.add_argument(
        '--target', 
        type=str, 
        default='101',
        help="要搜索的目标二进制字符串 (例如 '101')。"
    )
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=None,
        help="Grover 算法的迭代次数。如果未指定，将自动计算最佳次数。"
    )
    args = parser.parse_args()

    n_qubits = len(args.target)
    if n_qubits == 0:
        print("错误: 目标字符串不能为空。")
        return
    
    # 验证 target 是否为有效的二进制字符串
    if not all(c in '01' for c in args.target):
        print(f"错误: '{args.target}' 不是一个有效的二进制字符串。")
        return

    run_grover_search(args.target, args.iterations)


if __name__ == "__main__":
    main()