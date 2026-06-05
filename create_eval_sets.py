import os, random, pyarrow as pa, pyarrow.parquet as pq

random.seed(42)

def create_holdout(data_file, holdout_file, holdout_ratio=0.15, max_holdout=1000):
    if os.path.exists(holdout_file):
        print(f'SKIP {holdout_file} (exists, {os.path.getsize(holdout_file)} bytes)')
        return
    pf = pq.ParquetFile(data_file)
    total = pf.metadata.num_rows
    n = min(max(int(total * holdout_ratio), 100), max_holdout)
    print(f'  Reading {data_file} ({total} rows) -> holdout {n}')
    
    # Read all data
    table = pf.read()
    indices = sorted(random.sample(range(len(table)), n))
    holdout_table = table.take(indices)
    
    pq.write_table(holdout_table, holdout_file)
    print(f'CREATED {holdout_file}: {len(holdout_table)} rows ({os.path.getsize(holdout_file)/1024:.0f} KB)')

datasets = {
    'dataset/edu_ape210k.parquet': 'dataset/eval/eval_ape210k.parquet',
    'dataset/edu_chartqa.parquet': 'dataset/eval/eval_chartqa.parquet',
    'dataset/edu_cmmlu.parquet': 'dataset/eval/eval_cmmlu.parquet',
    'dataset/edu_math_verse.parquet': 'dataset/eval/eval_math_verse.parquet',
    'dataset/edu_math_vista.parquet': 'dataset/eval/eval_math_vista.parquet',
    'dataset/edu_ocr.parquet': 'dataset/eval/eval_ocr.parquet',
    'dataset/edu_openr1_math.parquet': 'dataset/eval/eval_openr1_math.parquet',
    'dataset/edu_race.parquet': 'dataset/eval/eval_race.parquet',
    'dataset/edu_science.parquet': 'dataset/eval/eval_science.parquet',
}

os.makedirs('dataset/eval', exist_ok=True)
for src, dst in datasets.items():
    if os.path.exists(src) and os.path.getsize(src) > 100:
        create_holdout(src, dst)
    else:
        print(f'SKIP {src} (not found)')

print('\nDone!')