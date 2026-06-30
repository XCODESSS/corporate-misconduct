import pyarrow.parquet as pq
import pyarrow.compute as pc

for name, path in [
    ("trainval", r"D:\coperate-misconduct-warning\data\processed\datasets\trainval_dataset.parquet"),
    ("test",     r"D:\coperate-misconduct-warning\data\processed\datasets\test_dataset.parquet"),
]:
    table = pq.read_table(path, columns=["fraudulent"])
    fraud = pc.sum(table["fraudulent"]).as_py()
    total = table.num_rows
    print(f"{name}: total={total} fraud={fraud} rate={fraud/total:.4f}")