import pandas as pd
import os
import numpy as np
import json
from configs import M_SEEN_MAX, M_SEEN_MIN, MU_SEEN_MAX, MU_SEEN_MIN, M_UNSEEN_MAX, MU_UNSEEN_MAX, GLOBAL_M_RANGE, GLOBAL_MU_RANGE, GLOBAL_FRIC_RANGE, REAL_M_RANGE, REAL_MU_RANGE, REAL_FRIC_RANGE, INCLUDE_UNSEEN


# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
OUTPUT_DIR = "results/paper/real"
G = 9.81  # Gravity constant for friction force calculation

# --- UPDATE THIS PATH TO MATCH YOUR TIMESTAMP RUN ---
# BASE_RUN_DIR = "/home/psxkf4/phypush_training/results/checkpoints/from_20260316"
BASE_RUN_DIR = "/home/psxkf4/PhyPush/results/checkpoints/from_20260618"


# --- REAL-WORLD FRICTION MAP ---
MU_MAP = {
    ("green_rub", "nolid_cube"): 0.4334,
    ("green_rub", "blue_cylinder"): 0.3654,
    ("green_rub", "colored_cubes"): 0.2987,
    ("green_rub", "wooden_cube"): 0.3103,
    ("green_rub", "wooden_cube_even"): 0.4626,
    ("wood_smooth", "nolid_cube"): 0.3421,
    ("wood_smooth", "blue_cylinder"): 0.2724,
    ("wood_smooth", "colored_cubes"): 0.1580,
    ("wood_smooth", "wooden_cube"): 0.1802,
    ("wood_smooth", "wooden_cube_even"): 0.2803,
    ("wood_rough", "nolid_cube"): 0.2550,
    ("wood_rough", "blue_cylinder"): 0.3747,
    ("wood_rough", "colored_cubes"): 0.1799,
    ("wood_rough", "wooden_cube"): 0.1621,
    ("wood_rough", "wooden_cube_even"): 0.1826,
}

csv_file = "real_evaluation_summary.csv"
# --- MANUAL MODEL SELECTION ---
MODELS_TO_COMPARE = {
    r"PhyPush": os.path.join(BASE_RUN_DIR, "20260619_202230", "pinn_pcri-L1_p5c10.0", csv_file),
    f"PropPush": "/home/psxkf4/CARD/phypush_diffusion/evaluation/results/real_evaluation_summary_proppush_20260619_202230.csv",
}

# --- METRICS ---
METRICS = {
    "mass_nrmse_pct": "mass_nrmse_pct",
    "mass_std_pct": "mass_std_pct",
    "best_mass_mean_err": "best_mass_mean_err",  
    "best_mass_est_std_dev": "best_mass_est_std_dev", 
    "best_mass_est_raw": "best_mass_est_raw",         
    "mu_nrmse_pct": "mu_nrmse_pct",
    "mu_std_pct": "mu_std_pct",
    "best_mu_mean_err": "best_mu_mean_err",
    "best_mu_est_std_dev": "best_mu_est_std_dev",     
    "best_mu_est_raw": "best_mu_est_raw",             
    "fric_f_nrmse_pct": "fric_f_nrmse_pct", 
    "fric_f_std_pct": "fric_f_std_pct",     
    "best_fric_f_mean_err": "best_fric_f_mean_err",
    "best_fric_f_est_std_dev": "best_fric_f_est_std_dev", 
    "best_fric_f_est_raw": "best_fric_f_est_raw"      
}

# ==========================================
# 2. PROCESSING FUNCTIONS
# ==========================================
def categorize_domain(domain_name):
    """
    Parses real-world domains like 'nolid_cube_0.498_wood_smooth'
    Returns Macro Domain ('Seen' or 'Unseen') and formatted strings for Obj, Surf, and GT.
    """
    parts = str(domain_name).split('_')
    
    if len(parts) >= 5:
        if "even" in domain_name:
            obj_type = f"{parts[0]}_{parts[1]}_even"
            mass_idx = 3
        else:
            obj_type = f"{parts[0]}_{parts[1]}"
            mass_idx = 2
            
        # OMIT WOODEN_CUBE (But keep wooden_cube_even)
        if obj_type == "wooden_cube":
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
            
        surface_type = f"{parts[-2]}_{parts[-1]}"
        
        obj_map = {
            "nolid_cube": "Plastic cube",
            "wooden_cube_even": "Wooden box",
            "blue_cylinder": "Tin Can",
            "colored_cubes": "Plastic container"
        }
        surf_map = {
            "green_rub": "Rubber",
            "wood_smooth": "Wood smooth",
            "wood_rough": "Wood rough"
        }
        
        obj_str = obj_map.get(obj_type, obj_type)
        surf_str = surf_map.get(surface_type, surface_type)
        
        try:
            mass = float(parts[mass_idx])
            mass_str = parts[mass_idx] 
            mu = MU_MAP.get((surface_type, obj_type), -1.0)
            
            mu_str = f"{mu:.4f}" if mu != -1.0 else "N/A"
            fric_str = f"{mass * mu * G:.2f}" if mu != -1.0 else "N/A"
            
            gt_str = f"{mass_str} / {mu_str} / {fric_str}"
            
            if (obj_type == "nolid_cube" and 
                (M_SEEN_MIN <= mass <= M_SEEN_MAX) and 
                (MU_SEEN_MIN <= mu <= MU_SEEN_MAX)):
                return pd.Series(['Seen', obj_str, surf_str, gt_str])
            else:
                return pd.Series(['Unseen', obj_str, surf_str, gt_str])
                
        except ValueError:
            pass
            
    return pd.Series(['Unseen', 'Unknown', 'Unknown', 'Unknown'])

def load_data(models_dict):
    combined_df = pd.DataFrame()
    for model_name, csv_path in models_dict.items():
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            df['Model'] = model_name
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    return combined_df

def generate_separated_reversed_tables(df, output_dir):
    if df.empty: return
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Map to macro domains and extract components
    df[['macro_domain', 'Object', 'Surface', 'GT']] = df['domain'].apply(categorize_domain)
    
    # Drop rows flagged as NaN (like wooden_cube)
    df.dropna(subset=['macro_domain'], inplace=True)
    
    df['combo'] = df['Object'] + " | " + df['Surface'] + " | " + df['GT']
    
    for m in METRICS.values():
        if m not in df.columns:
            df[m] = np.nan
            
    # 2. Pivot so Combos are rows and Models are columns
    agg_df = df.groupby(['macro_domain', 'combo', 'Object', 'Surface', 'GT', 'Model'], observed=False)[list(METRICS.values())].mean().reset_index()
    pivot_df = agg_df.pivot(index=['macro_domain', 'combo', 'Object', 'Surface', 'GT'], columns='Model', values=list(METRICS.values()))
    
    model_order = list(MODELS_TO_COMPARE.keys())
    valid_models = [m for m in model_order if m in pivot_df.columns.get_level_values('Model')]
    
    # 3. Find the minimum values across rows to apply Bold Formatting
    best_nrmse = {}
    best_bias = {}
    
    for idx in pivot_df.index:
        for prop in ['mass', 'mu', 'fric_f']:
            # Best NRMSE is the minimum value across models for this row
            nrmse_cols = [(f'{prop}_nrmse_pct', m) for m in valid_models if (f'{prop}_nrmse_pct', m) in pivot_df.columns]
            best_nrmse[(idx, prop)] = pivot_df.loc[idx, nrmse_cols].min() if nrmse_cols else np.nan
            
            # Best Bias is the minimum absolute value across models (closest to 0)
            bias_cols = [(f'best_{prop}_mean_err', m) for m in valid_models if (f'best_{prop}_mean_err', m) in pivot_df.columns]
            if bias_cols:
                biases = pivot_df.loc[idx, bias_cols].values
                valid_biases = [b for b in biases if pd.notna(b)]
                best_bias[(idx, prop)] = min(np.abs(valid_biases)) if valid_biases else np.nan
            else:
                best_bias[(idx, prop)] = np.nan

    # =========================================================
    # REVERSED FORMAT BUILDER (TWO TABLES)
    # =========================================================
    latex_rows_nrmse = []
    csv_rows_nrmse = []
    
    latex_rows_bias = []
    csv_rows_bias = []
    
    for g in ['Seen', 'Unseen']:
        group_idx = [idx for idx in pivot_df.index if idx[0] == g]
        group_idx = sorted(group_idx, key=lambda x: x[1]) 
        
        if not group_idx: continue
            
        num_cols = 3 + (len(valid_models) * 3) 
        if g == 'Seen':
            header_str = f"        \\midrule\n        \\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{g} Object}}}} \\\\\n        \\midrule"
        else:
            header_str = f"        \\midrule\n        \\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{g} Objects}}}} \\\\\n        \\midrule"
        latex_rows_nrmse.append(header_str)
        latex_rows_bias.append(header_str)
        
        for idx in group_idx:
            row_data = pivot_df.loc[idx]
            _, _, obj_str, surf_str, gt_str = idx
            
            tex_cols_nrmse = [obj_str, surf_str, gt_str]
            csv_cols_nrmse = [g, obj_str, surf_str, gt_str]
            
            tex_cols_bias = [obj_str, surf_str, gt_str]
            csv_cols_bias = [g, obj_str, surf_str, gt_str]
            
            for model_name in valid_models:
                # --- NRMSE ---
                for prop in ['mass', 'mu', 'fric_f']:
                    nrmse_col = (f"{prop}_nrmse_pct", model_name)
                    std_col = (f"{prop}_std_pct", model_name)
                    
                    nrmse_val = row_data.get(nrmse_col, np.nan)
                    std_val = row_data.get(std_col, np.nan)
                    
                    if pd.isna(nrmse_val):
                        tex_cols_nrmse.append("-")
                        csv_cols_nrmse.append("-")
                    else:
                        nrmse_val = nrmse_val / 100.0
                        std_val = std_val / 100.0
                        
                        std_str = f"{std_val:.3f}" if pd.notna(std_val) else "N/A"
                        
                        best_nrmse_raw = best_nrmse[(idx, prop)] / 100.0
                        is_best = abs(nrmse_val - best_nrmse_raw) < 1e-6
                        
                        if is_best:
                            tex_cols_nrmse.append(f"\\textbf{{{nrmse_val:.3f}}}$\\pm${std_str}")
                        else:
                            tex_cols_nrmse.append(f"{nrmse_val:.3f}$\\pm${std_str}")
                        csv_cols_nrmse.append(f"{nrmse_val:.3f}±{std_str}")

                # --- EST ± ESTIMATION_STD (RAW_ERROR) ---
                for prop in ['mass', 'mu', 'fric_f']:
                    bias_col = (f"best_{prop}_mean_err", model_name)
                    est_std_col = (f"best_{prop}_est_std_dev", model_name) 
                    est_raw_col = (f"best_{prop}_est_raw", model_name) 
                    
                    bias_val = row_data.get(bias_col, np.nan)
                    est_std_val = row_data.get(est_std_col, np.nan)
                    est_raw_val = row_data.get(est_raw_col, np.nan)
                    
                    if pd.isna(bias_val):
                        tex_cols_bias.append("-")
                        csv_cols_bias.append("-")
                    else:
                        bias_str = f"{bias_val:+.3f}"
                        est_std_str = f"{est_std_val:.3f}" if pd.notna(est_std_val) else "N/A"
                        est_raw_str = f"{est_raw_val:.3f}" if pd.notna(est_raw_val) else "N/A"
                        
                        is_best = abs(abs(bias_val) - best_bias[(idx, prop)]) < 1e-6
                        
                        if is_best:
                            tex_cols_bias.append(f"\\textbf{{{est_raw_str}}}$\\pm${est_std_str}({bias_str})")
                        else:
                            tex_cols_bias.append(f"{est_raw_str}$\\pm${est_std_str}({bias_str})")
                        
                        csv_cols_bias.append(f"{est_raw_str}±{est_std_str}({bias_str})")

            latex_rows_nrmse.append("        " + " & ".join(tex_cols_nrmse) + " \\\\")
            csv_rows_nrmse.append(csv_cols_nrmse)
            
            latex_rows_bias.append("        " + " & ".join(tex_cols_bias) + " \\\\")
            csv_rows_bias.append(csv_cols_bias)

    # =========================================================
    # WRITE OUTPUT FILES
    # =========================================================
    csv_header_nrmse = ["Category", "Object", "Surface", "GT (m / mu / Fric)"]
    csv_header_bias = ["Category", "Object", "Surface", "GT (m / mu / Fric)"]
    for m in valid_models:
        csv_header_nrmse.extend([f"{m} (m NRMSE±STD)", f"{m} (mu NRMSE±STD)", f"{m} (Fric F NRMSE±STD)"])
        csv_header_bias.extend([f"{m} (m Est±Est_STD(Bias))", f"{m} (mu Est±Est_STD(Bias))", f"{m} (Fric F Est±Est_STD(Bias))"])
        
    pd.DataFrame(csv_rows_nrmse, columns=csv_header_nrmse).to_csv(os.path.join(output_dir, "real_nrmse_reversed.csv"), index=False)
    pd.DataFrame(csv_rows_bias, columns=csv_header_bias).to_csv(os.path.join(output_dir, "real_bias_reversed.csv"), index=False)
    
    r1 = "        \\multirow{2}{*}{\\textbf{Object}} & \\multirow{2}{*}{\\textbf{Surface}} & \\multirow{2}{*}{\\textbf{GT ($m$ / $\\mu$ / $F_{\\text{fric}}$)}} "
    for m in valid_models:
        safe_m = str(m).replace('_', r'\_') if '$' not in str(m) else str(m)
        r1 += f"& \\multicolumn{{3}}{{c}}{{\\textbf{{{safe_m}}}}} " 
    r1 += "\\\\"
    
    cmid1 = "        "
    curr = 4
    for _ in valid_models:
        cmid1 += f"\\cmidrule(lr){{{curr}-{curr + 2}}} " 
        curr += 3
        
    r2_cols = ["\\textbf{$m$}", "\\textbf{$\\mu$}", "\\textbf{$F_{\\text{fric}}$}"] * len(valid_models) 
    r2 = "        & & & " + " & ".join(r2_cols) + " \\\\"

    tex_path_nrmse = os.path.join(output_dir, "real_nrmse_reversed.tex")
    latex_table_nrmse = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Real-World Performance Comparison (NRMSE $\\pm$ STD) by Object}",
        "    \\vspace{2mm}",
        "    \\resizebox{\\textwidth}{!}{",
        f"    \\begin{{tabular}}{{l l c {'c c c ' * len(valid_models)}}}", 
        "        \\toprule",
        r1, cmid1, r2
    ]
    latex_table_nrmse.extend(latex_rows_nrmse)
    latex_table_nrmse.extend(["        \\bottomrule", "    \\end{tabular}", "    }", "\\end{table*}"])
    with open(tex_path_nrmse, "w") as f:
        f.write("\n".join(latex_table_nrmse) + "\n")

    tex_path_bias = os.path.join(output_dir, "real_bias_reversed.tex")
    latex_table_bias = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Real-World Performance Comparison (Top-5 Est $\\pm$ Est STD (Raw Error)) by Object}",
        "    \\vspace{2mm}",
        "    \\resizebox{\\textwidth}{!}{",
        f"    \\begin{{tabular}}{{l l c {'c c c ' * len(valid_models)}}}", 
        "        \\toprule",
        r1, cmid1, r2
    ]
    latex_table_bias.extend(latex_rows_bias)
    latex_table_bias.extend(["        \\bottomrule", "    \\end{tabular}", "    }", "\\end{table*}"])
    with open(tex_path_bias, "w") as f:
        f.write("\n".join(latex_table_bias) + "\n")

    print(f"Generated NRMSE Format (Reversed): {tex_path_nrmse}")
    print(f"Generated Bias Format (Reversed): {tex_path_bias}")
    print(f"\nAll tasks completed. Results in: {OUTPUT_DIR}")

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print(f"Processing Separated Reversed Real-World Tables (with Est±Est_STD(Bias))...")
    df_results = load_data(MODELS_TO_COMPARE)
    generate_separated_reversed_tables(df_results, OUTPUT_DIR)