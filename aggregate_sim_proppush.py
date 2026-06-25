import pandas as pd
import os
import numpy as np
from configs import INCLUDE_UNSEEN

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
OUTPUT_DIR = "results/paper/sim"
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")
PHYSICS_DIR = os.path.join(OUTPUT_DIR, "physics_fidelity")


BASE_RUN_DIR = "/home/psxkf4/PhyPush/results/checkpoints/from_20260618"

csv_file = "domain_evaluation_summary.csv"
MODELS_TO_COMPARE = {
    r"Baseline~\cite{mavrakis_estimating_2020}": {
        "path": f"/home/psxkf4/phypush_training/results/checkpoints/baseline_random_forest/{csv_file}",
        "force_input": "Yes"
    },
    r"PhyPush open gripper": {
        "path": os.path.join(BASE_RUN_DIR, "20260619_202230", f"pinn_pcri-L1_p5c10.0/{csv_file}"),
        "force_input": "No"
    },

    r"PhyPush closed gripper": {
        "path": os.path.join(BASE_RUN_DIR, "", f"pinn_pcri-L1_p5c10.0/{csv_file}"),
        "force_input": "No"
    },

    r"PropPush open gripper": {
        "path": "/home/psxkf4/CARD/phypush_diffusion/evaluation/results/domain_evaluation_summary_proppush_20260619_202230.csv",
        "force_input": "No"
    },

    # r"PropPush closed gripper": {
    #     "path": "/home/psxkf4/CARD/phypush_diffusion/evaluation/results/domain_evaluation_summary_proppush_20260625_005028.csv",
    #     "force_input": "No"
    # },

}

# --- METRICS DEFINITIONS ---
METRICS_NRMSE = {
    "mass_nrmse_pct": "mass_nrmse_pct",
    "mass_std_pct": "mass_std_pct",
    "mu_nrmse_pct": "mu_nrmse_pct",
    "mu_std_pct": "mu_std_pct",
}

METRICS_ABLATION = {
    "mass_nrmse_pct": "mass_nrmse_pct",
    "mass_std_pct": "mass_std_pct",
    "mass_mean_err": "mass_mean_err",  
    "mass_std_dev": "mass_std_dev",    
    "mu_nrmse_pct": "mu_nrmse_pct",
    "mu_std_pct": "mu_std_pct",
    "mu_mean_err": "mu_mean_err",      
    "mu_std_dev": "mu_std_dev"         
}

METRICS_R2 = {
    "net_f_r2": "net_f_r2",
    "fric_r2": "fric_r2"
}

# --- MACRO-DOMAIN MAPPINGS ---
if INCLUDE_UNSEEN:
    DOMAIN_GROUPS_MAIN = {
        r'$\mathcal{D}_{test}$': ['m_seen_mu_seen'],
        r'$\mathcal{D}_{OOD}^m$': ['m_over'],
        r'$\mathcal{D}_{OOD}^{\mu}$': ['mu_over'],
        r'$\mathcal{D}_{OOD}^{m, \mu}$': ['m_over_mu_over']
    }
    DOMAIN_GROUPS_ABLATION = {
        r'$\mathcal{D}_{test}$': ['m_seen_mu_seen'],
        r'{\scriptsize $\mathcal{D}_{OOD}^m$ $\cup$ $\mathcal{D}_{OOD}^{\mu}$ $\cup$ $\mathcal{D}_{OOD}^{m, \mu}$}': [
            'm_over', 'm_under', 'mu_over', 'mu_under',
            'm_over_mu_over', 'm_under_mu_under', 'm_over_mu_under', 'm_under_mu_over'
        ]
    }
    ALL_MAPPINGS_R2 = {
        "_separated": DOMAIN_GROUPS_MAIN,
        "_combined": DOMAIN_GROUPS_ABLATION
    }
else:
    DOMAIN_GROUPS_MAIN = {
        'Light (0.2~0.8 kg)': ['m_seen_light'],
        'Middle (0.8~1.4 kg)': ['m_seen_middle'],
        'Heavy (1.4~2.0 kg)': ['m_seen_heavy']
    }
    DOMAIN_GROUPS_ABLATION = DOMAIN_GROUPS_MAIN
    ALL_MAPPINGS_R2 = { "": DOMAIN_GROUPS_MAIN }


# ==========================================
# 2. SHARED UTILITIES
# ==========================================
def load_data(models_dict, key_is_dict=True):
    combined_df = pd.DataFrame()
    for model_name, info in models_dict.items():
        csv_path = info["path"] if key_is_dict else info
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

# ==========================================
# 3. TASK 1: MAIN NRMSE SIMULATION TABLES
# ==========================================
def generate_main_tables(df, output_dir):
    if df.empty: return
    os.makedirs(output_dir, exist_ok=True)
    
    mapping = {}
    for grp, doms in DOMAIN_GROUPS_MAIN.items():
        for d in doms:
            mapping[d] = grp
            
    df['macro_domain'] = df['domain'].map(mapping)
    df = df.dropna(subset=['macro_domain']) 
    
    metrics_cols = list(METRICS_NRMSE.values())
    for m in metrics_cols:
        if m not in df.columns:
            df[m] = np.nan
            
    agg_df = df.groupby(['Model', 'macro_domain'], observed=False)[metrics_cols].mean().reset_index()
    pivot_df = agg_df.pivot(index='Model', columns='macro_domain', values=metrics_cols)
    
    group_order = list(DOMAIN_GROUPS_MAIN.keys())
    model_order = list(MODELS_TO_COMPARE.keys())
    
    overall_name = "Overall"
    groups_to_average = [g for g in group_order if 'OOD' in g] if INCLUDE_UNSEEN else group_order.copy()
        
    for prop in ['mass', 'mu']:
        for met_base in ['nrmse_pct', 'std_pct']:
            full_col_name = f"{prop}_{met_base}"
            vals_to_avg = []
            for g in groups_to_average:
                if (full_col_name, g) in pivot_df.columns:
                    vals_to_avg.append(pivot_df[(full_col_name, g)])
            
            if vals_to_avg:
                pivot_df[(full_col_name, overall_name)] = pd.concat(vals_to_avg, axis=1).mean(axis=1)

    group_order.append(overall_name)

    best_nrmse = {}
    for g in group_order:
        for prop in ['mass', 'mu']:
            nrmse_col = (f'{prop}_nrmse_pct', g)
            best_nrmse[(prop, g)] = pivot_df[nrmse_col].min() if nrmse_col in pivot_df.columns else np.nan

    latex_rows_nrmse_narrow, csv_rows_nrmse_narrow = [], []
    valid_models = [m for m in model_order if m in pivot_df.index]
    
    num_no_models = sum(1 for m in valid_models if MODELS_TO_COMPARE[m]["force_input"] == "No")
    phypush_rows_total = num_no_models * 2
    first_no_seen = False
    
    for idx, model_name in enumerate(valid_models):
        row_data = pivot_df.loc[model_name]
        force_input = MODELS_TO_COMPARE[model_name]["force_input"]
        safe_model_name = str(model_name)
        
        if force_input == "Yes":
            cmark_str = r"\multirow{2}{*}{\textbf{\cmark}}"
        else:
            if not first_no_seen:
                cmark_str = rf"\multirow{{{phypush_rows_total}}}{{*}}{{\textbf{{\xmark}}}}"
                first_no_seen = True
            else:
                cmark_str = ""
        
        tex_cols_n_s_m = [f"\\multirow{{2}}{{*}}{{{safe_model_name}}}", cmark_str, "$m$"]
        csv_cols_n_s_m = [model_name, force_input, "$m$"]
        tex_cols_n_s_mu = ["", "", "$\\mu$"]
        csv_cols_n_s_mu = ["", "", "$\\mu$"]
        
        for g in group_order:
            m_nrmse = row_data.get(('mass_nrmse_pct', g), np.nan)
            m_std_pct = row_data.get(('mass_std_pct', g), np.nan)
            
            if pd.isna(m_nrmse):
                tex_cols_n_s_m.append("-")
                csv_cols_n_s_m.append("-")
            else:
                m_nrmse = m_nrmse / 100.0
                m_std_pct = m_std_pct / 100.0
                is_best = abs(m_nrmse - (best_nrmse[('mass', g)] / 100.0)) < 1e-6
                std_str = f"{m_std_pct:.3f}" if pd.notna(m_std_pct) else "N/A"
                str_s = f"\\textbf{{{m_nrmse:.3f}}}$\\pm${std_str}" if is_best else f"{m_nrmse:.3f}$\\pm${std_str}"
                tex_cols_n_s_m.append(str_s)
                csv_cols_n_s_m.append(f"{m_nrmse:.3f}±{std_str}")

            mu_nrmse = row_data.get(('mu_nrmse_pct', g), np.nan)
            mu_std_pct = row_data.get(('mu_std_pct', g), np.nan)
            
            if pd.isna(mu_nrmse):
                tex_cols_n_s_mu.append("-")
                csv_cols_n_s_mu.append("-")
            else:
                mu_nrmse = mu_nrmse / 100.0
                mu_std_pct = mu_std_pct / 100.0
                is_best = abs(mu_nrmse - (best_nrmse[('mu', g)] / 100.0)) < 1e-6
                std_str = f"{mu_std_pct:.3f}" if pd.notna(mu_std_pct) else "N/A"
                str_s = f"\\textbf{{{mu_nrmse:.3f}}}$\\pm${std_str}" if is_best else f"{mu_nrmse:.3f}$\\pm${std_str}"
                tex_cols_n_s_mu.append(str_s)
                csv_cols_n_s_mu.append(f"{mu_nrmse:.3f}±{std_str}")

        cols_span = str(len(tex_cols_n_s_m))
        latex_rows_nrmse_narrow.append("        " + " & ".join(tex_cols_n_s_m) + r" \\")
        latex_rows_nrmse_narrow.append(rf"        \cmidrule(lr){{3-{cols_span}}}")
        latex_rows_nrmse_narrow.append("        " + " & ".join(tex_cols_n_s_mu) + r" \\")
        
        if idx < len(valid_models) - 1:
            next_model = valid_models[idx + 1]
            if force_input == "Yes" and MODELS_TO_COMPARE[next_model]["force_input"] == "No":
                latex_rows_nrmse_narrow.append(r"        \midrule")
                latex_rows_nrmse_narrow.append(r"        \midrule")
            elif force_input == "No" and MODELS_TO_COMPARE[next_model]["force_input"] == "No":
                latex_rows_nrmse_narrow.append(rf"        \cmidrule(lr){{3-{cols_span}}}")
            else:
                latex_rows_nrmse_narrow.append(r"        \midrule")
            
        csv_rows_nrmse_narrow.extend([csv_cols_n_s_m, csv_cols_n_s_mu])

    csv_header_narrow = ["Model", "Force Input Required?", "Property"] + group_order
    pd.DataFrame(csv_rows_nrmse_narrow, columns=csv_header_narrow).to_csv(os.path.join(output_dir, "macro_domains_nrmse.csv"), index=False)
    
    path = os.path.join(output_dir, "macro_domains_nrmse.tex")
    with open(path, "w") as f:
        f.write(r"\begin{table*}[tb]" + "\n")
        f.write(r"    \centering" + "\n")
        f.write(r"    \caption{Simulation (NRMSE $\pm$ std) ($\downarrow$) for Mass ($m$) and Friction ($\mu$) for seen and unseen domains. \textbf{\cmark} Requires force/torque input. \textbf{\xmark} Uses only kinematic velocity.\Done}" + "\n")
        f.write(r"    \begin{tabular}{l|c| c| c| c c c c}" + "\n")
        f.write(r"        \toprule" + "\n")
        if INCLUDE_UNSEEN:
            f.write(r"        \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Force ?}} & \multirow{2}{*}{\textbf{Property}} & \textbf{Seen} & \multicolumn{4}{c}{\textbf{Unseen}}\\" + "\n")
            f.write(r"        \cmidrule(lr){4-8}" + "\n")
            f.write(r"        & & &$\mathcal{D}_{test}$ & $\mathcal{D}_{OOD}^m$ & $\mathcal{D}_{OOD}^{\mu}$ & $\mathcal{D}_{OOD}^{m, \mu}$ & Overall \\" + "\n")
        else:
            f.write(r"        \textbf{Model} & \textbf{Force ?} & \textbf{Property} & " + " & ".join(group_order) + r" \\" + "\n")
        f.write(r"        \midrule" + "\n")
        f.write("\n".join(latex_rows_nrmse_narrow) + "\n")
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}\label{tab:sim_nrmse}" + "\n")
        f.write(r"\end{table*}" + "\n")

    print(f"Generated Main NRMSE tables in: {output_dir}")


# ==========================================
# 4. TASK 2: ABLATION TABLES
# ==========================================
def generate_ablation_tables(df, output_dir):
    if df.empty: return
    os.makedirs(output_dir, exist_ok=True)
    
    mapping = {}
    for grp, doms in DOMAIN_GROUPS_ABLATION.items():
        for d in doms:
            mapping[d] = grp
            
    df['macro_domain'] = df['domain'].map(mapping)
    df = df.dropna(subset=['macro_domain']) 
    
    metrics_cols = list(METRICS_ABLATION.values())
    for m in metrics_cols:
        if m not in df.columns:
            df[m] = np.nan
            
    agg_df = df.groupby(['Model', 'macro_domain'], observed=False)[metrics_cols].mean().reset_index()
    pivot_df = agg_df.pivot(index='Model', columns='macro_domain', values=metrics_cols)
    
    group_order = list(DOMAIN_GROUPS_ABLATION.keys())
    model_order = list(MODELS_TO_COMPARE.keys())

    for prop in ['mass', 'mu']:
        for met_base in ['nrmse_pct', 'std_pct', 'mean_err', 'std_dev']:
            full_col_name = f"{prop}_{met_base}"
            vals_to_avg = []
            for g in group_order:
                if (full_col_name, g) in pivot_df.columns:
                    vals_to_avg.append(pivot_df[(full_col_name, g)])
            if vals_to_avg:
                pivot_df[(full_col_name, 'Overall')] = pd.concat(vals_to_avg, axis=1).mean(axis=1)

    group_order.append('Overall')
    
    for m_name in pivot_df.index:
        if "Only $m$" in m_name:
            for g in group_order:
                for met in ['mu_nrmse_pct', 'mu_std_pct', 'mu_mean_err', 'mu_std_dev']:
                    if (met, g) in pivot_df.columns:
                        pivot_df.loc[m_name, (met, g)] = np.nan

        if "Only $\\mu$" in m_name:
            for g in group_order:
                for met in ['mass_nrmse_pct', 'mass_std_pct', 'mass_mean_err', 'mass_std_dev']:
                    if (met, g) in pivot_df.columns:
                        pivot_df.loc[m_name, (met, g)] = np.nan

    best_nrmse = {}
    best_bias = {}
    for g in group_order:
        for prop in ['mass', 'mu']:
            nrmse_col = (f'{prop}_nrmse_pct', g)
            best_nrmse[(prop, g)] = pivot_df[nrmse_col].min() if nrmse_col in pivot_df.columns else np.nan
            
            bias_col = (f'{prop}_mean_err', g)
            if bias_col in pivot_df.columns:
                biases = pivot_df[bias_col].dropna()
                best_bias[(prop, g)] = biases.abs().min() if len(biases) > 0 else np.nan
            else:
                best_bias[(prop, g)] = np.nan

    latex_rows_nrmse_wide, csv_rows_nrmse_wide = [], []
    latex_rows_bias_wide, csv_rows_bias_wide = [], []
    latex_rows_nrmse_narrow, csv_rows_nrmse_narrow = [], []
    latex_rows_bias_narrow, csv_rows_bias_narrow = [], []
    
    for idx, model_name in enumerate(model_order):
        if model_name not in pivot_df.index: continue
            
        row_data = pivot_df.loc[model_name]
        safe_model_name = str(model_name)
        
        tex_cols_n_w, csv_cols_n_w = [safe_model_name], [model_name]
        tex_cols_b_w, csv_cols_b_w = [safe_model_name], [model_name]
        
        tex_cols_n_s_m, csv_cols_n_s_m = [f"\\multirow{{2}}{{*}}{{{safe_model_name}}}", "$m$"], [model_name, "$m$"]
        tex_cols_n_s_mu, csv_cols_n_s_mu = ["", "$\\mu$"], ["", "$\\mu$"]
        
        tex_cols_b_s_m, csv_cols_b_s_m = [f"\\multirow{{2}}{{*}}{{{safe_model_name}}}", "$m$"], [model_name, "$m$"]
        tex_cols_b_s_mu, csv_cols_b_s_mu = ["", "$\\mu$"], ["", "$\\mu$"]

        for prop in ['mass', 'mu']:
            for g in group_order:
                nrmse_val = row_data.get((f'{prop}_nrmse_pct', g), np.nan)
                std_pct_val = row_data.get((f'{prop}_std_pct', g), np.nan)
                bias_val = row_data.get((f'{prop}_mean_err', g), np.nan)
                std_raw_val = row_data.get((f'{prop}_std_dev', g), np.nan)

                if pd.isna(nrmse_val):
                    tex_cols_n_w.append("N/A"); csv_cols_n_w.append("N/A")
                    if prop == 'mass':
                        tex_cols_n_s_m.append("N/A"); csv_cols_n_s_m.append("N/A")
                    else:
                        tex_cols_n_s_mu.append("N/A"); csv_cols_n_s_mu.append("N/A")
                else:
                    nrmse_val = nrmse_val / 100.0
                    std_pct_val = std_pct_val / 100.0
                    best_nrmse_raw = best_nrmse[(prop, g)] / 100.0
                    is_best = abs(nrmse_val - best_nrmse_raw) < 1e-6
                    std_str = f"{std_pct_val:.3f}" if pd.notna(std_pct_val) else "N/A"
                    
                    str_n_w = f"\\textbf{{{nrmse_val:.3f}}} $\\pm$ {std_str}" if is_best else f"{nrmse_val:.3f} $\\pm$ {std_str}"
                    tex_cols_n_w.append(str_n_w); csv_cols_n_w.append(f"{nrmse_val:.3f}±{std_str}")
                    
                    str_n_s = f"\\textbf{{{nrmse_val:.3f}}}$\\pm${std_str}" if is_best else f"{nrmse_val:.3f}$\\pm${std_str}"
                    if prop == 'mass':
                        tex_cols_n_s_m.append(str_n_s); csv_cols_n_s_m.append(f"{nrmse_val:.3f}±{std_str}")
                    else:
                        tex_cols_n_s_mu.append(str_n_s); csv_cols_n_s_mu.append(f"{nrmse_val:.3f}±{std_str}")

                if pd.isna(bias_val):
                    tex_cols_b_w.append("N/A"); csv_cols_b_w.append("N/A")
                    if prop == 'mass':
                        tex_cols_b_s_m.append("N/A"); csv_cols_b_s_m.append("N/A")
                    else:
                        tex_cols_b_s_mu.append("N/A"); csv_cols_b_s_mu.append("N/A")
                else:
                    is_best = abs(abs(bias_val) - best_bias[(prop, g)]) < 1e-6
                    std_raw = f"{std_raw_val:.3f}" if pd.notna(std_raw_val) else "N/A"
                    
                    str_b_w = f"\\textbf{{{bias_val:+.3f}}} $\\pm$ {std_raw}" if is_best else f"{bias_val:+.3f} $\\pm$ {std_raw}"
                    tex_cols_b_w.append(str_b_w); csv_cols_b_w.append(f"{bias_val:+.3f}±{std_raw}")
                    
                    str_b_s = f"\\textbf{{{bias_val:+.3f}}}$\\pm${std_raw}" if is_best else f"{bias_val:+.3f}$\\pm${std_raw}"
                    if prop == 'mass':
                        tex_cols_b_s_m.append(str_b_s); csv_cols_b_s_m.append(f"{bias_val:+.3f}±{std_raw}")
                    else:
                        tex_cols_b_s_mu.append(str_b_s); csv_cols_b_s_mu.append(f"{bias_val:+.3f}±{std_raw}")

        latex_rows_nrmse_wide.append("        " + " & ".join(tex_cols_n_w) + " \\\\")
        csv_rows_nrmse_wide.append(csv_cols_n_w)
        latex_rows_bias_wide.append("        " + " & ".join(tex_cols_b_w) + " \\\\")
        csv_rows_bias_wide.append(csv_cols_b_w)
        
        cols_span = str(len(tex_cols_n_s_m))
        latex_rows_nrmse_narrow.append("        " + " & ".join(tex_cols_n_s_m) + " \\\\")
        latex_rows_nrmse_narrow.append(f"        \\cmidrule(lr){{2-{cols_span}}}")
        latex_rows_nrmse_narrow.append("        " + " & ".join(tex_cols_n_s_mu) + " \\\\")
        
        latex_rows_bias_narrow.append("        " + " & ".join(tex_cols_b_s_m) + " \\\\")
        latex_rows_bias_narrow.append(f"        \\cmidrule(lr){{2-{cols_span}}}")
        latex_rows_bias_narrow.append("        " + " & ".join(tex_cols_b_s_mu) + " \\\\")
        
        if idx < len(model_order) - 1:
            latex_rows_nrmse_narrow.append("        \\midrule")
            latex_rows_bias_narrow.append("        \\midrule")
            
        csv_rows_nrmse_narrow.extend([csv_cols_n_s_m, csv_cols_n_s_mu])
        csv_rows_bias_narrow.extend([csv_cols_b_s_m, csv_cols_b_s_mu])

    csv_header_w = ["Model"] + [f"Mass ({g})" for g in group_order] + [f"Fric ({g})" for g in group_order]
    csv_header_s = ["Model", "Property"] + group_order
        
    pd.DataFrame(csv_rows_nrmse_wide, columns=csv_header_w).to_csv(os.path.join(output_dir, "ablation_nrmse_twocolumn.csv"), index=False)
    pd.DataFrame(csv_rows_bias_wide, columns=csv_header_w).to_csv(os.path.join(output_dir, "ablation_bias_twocolumn.csv"), index=False)
    pd.DataFrame(csv_rows_nrmse_narrow, columns=csv_header_s).to_csv(os.path.join(output_dir, "ablation_nrmse_onecolumn.csv"), index=False)
    pd.DataFrame(csv_rows_bias_narrow, columns=csv_header_s).to_csv(os.path.join(output_dir, "ablation_bias_onecolumn.csv"), index=False)
    
    num_g = len(group_order)
    r1_w = f"        \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & \\multicolumn{{{num_g}}}{{c}}{{\\textbf{{$m$}}}} & \\multicolumn{{{num_g}}}{{c}}{{\\textbf{{$\\mu$}}}} \\\\"
    r2_w = f"        \\cmidrule(lr){{2-{1+num_g}}} \\cmidrule(lr){{{2+num_g}-{1+2*num_g}}}\n        & " + " & ".join(group_order) + " & " + " & ".join(group_order) + " \\\\"
    r1_s = f"        \\textbf{{Model}} & \\textbf{{Property}} & " + " & ".join(group_order) + " \\\\"

    def write_tex(filename, title, latex_rows, is_wide):
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write("\\begin{table*}[t]\n" if is_wide else "\\begin{table}[htbp]\n")
            f.write("    \\centering\n")
            f.write(f"    \\caption{{{title}}}\n")
            f.write("    \\vspace{2mm}\n")
            f.write("    \\resizebox{0.8\\textwidth}{!}{\n" if is_wide else "    \\resizebox{\\columnwidth}{!}{\n")
            
            align_w = "l " + " ".join(["c" for _ in range(num_g * 2)])
            align_s = "l c " + " ".join(["c" for _ in range(num_g)])
            
            f.write(f"    \\begin{{tabular}}{{{align_w}}}\n" if is_wide else f"    \\begin{{tabular}}{{{align_s}}}\n")
            f.write("        \\toprule\n")
            if is_wide:
                f.write(f"{r1_w}\n{r2_w}\n")
            else:
                f.write(f"{r1_s}\n")
            f.write("        \\midrule\n")
            f.write("\n".join(latex_rows) + "\n")
            f.write("        \\bottomrule\n")
            f.write("    \\end{tabular}\n    }\n")
            f.write("\\end{table*}\n" if is_wide else "\\end{table}\n")

    write_tex("ablation_nrmse_twocolumn.tex", "Ablation Study: Performance (NRMSE $\\pm$ STD) ($\\downarrow$)", latex_rows_nrmse_wide, True)
    write_tex("ablation_bias_twocolumn.tex", "Ablation Study: Performance (Raw Bias $\\pm$ Error STD) ($\\downarrow$)", latex_rows_bias_wide, True)
    write_tex("ablation_nrmse_onecolumn.tex", "Ablation Study: Performance (NRMSE $\\pm$ STD) ($\\downarrow$)", latex_rows_nrmse_narrow, False)
    write_tex("ablation_bias_onecolumn.tex", "Ablation Study: Performance (Raw Bias $\\pm$ Error STD) ($\\downarrow$)", latex_rows_bias_narrow, False)

    print(f"Generated Ablation tables in: {output_dir}")


# ==========================================
# 5. TASK 3: R2 PHYSICS TABLES
# ==========================================
def generate_r2_tables(df, domain_groups, suffix, output_dir):
    if df.empty: return
    os.makedirs(output_dir, exist_ok=True)
    
    mapping = {}
    for grp, doms in domain_groups.items():
        for d in doms:
            mapping[d] = grp
            
    df_mapped = df.copy()
    df_mapped['macro_domain'] = df_mapped['domain'].map(mapping)
    df_mapped = df_mapped.dropna(subset=['macro_domain']) 
    
    metrics_cols = list(METRICS_R2.values())
    for m in metrics_cols:
        if m not in df_mapped.columns:
            df_mapped[m] = np.nan
            
    agg_df = df_mapped.groupby(['Model', 'macro_domain'], observed=False)[metrics_cols].mean().reset_index()
    pivot_df = agg_df.pivot(index='Model', columns='macro_domain', values=metrics_cols)
    
    group_order = list(domain_groups.keys())
    model_order = list(MODELS_TO_COMPARE.keys())
    
    overall_name = "Overall"
    for prop in ['net_f_r2', 'fric_r2']:
        vals_to_avg = []
        for g in group_order:
            if (prop, g) in pivot_df.columns:
                vals_to_avg.append(pivot_df[(prop, g)])
        if vals_to_avg:
            pivot_df[(prop, overall_name)] = pd.concat(vals_to_avg, axis=1).mean(axis=1)

    group_order.append(overall_name)

    best_r2 = {}
    for g in group_order:
        for prop in ['net_f_r2', 'fric_r2']:
            r2_col = (prop, g)
            best_r2[(prop, g)] = pivot_df[r2_col].max() if r2_col in pivot_df.columns else np.nan

    latex_rows_r2_narrow, csv_rows_r2_narrow = [], []
    valid_models = [m for m in model_order if m in pivot_df.index]
    
    for idx, model_name in enumerate(valid_models):
        row_data = pivot_df.loc[model_name]
        safe_model_name = str(model_name)
        
        tex_cols_s_net = [f"\\multirow{{2}}{{*}}{{{safe_model_name}}}", r"$F_{\text{net}}=\widehat{m}a_{\text{obj}}$"]
        csv_cols_s_net = [model_name, "Net F"]
        tex_cols_s_fric = ["", r"$F_{\text{fric}}=\widehat{\mu}\widehat{m}g$"]
        csv_cols_s_fric = ["", "Fric F"]
        
        for g in group_order:
            net_val = row_data.get(('net_f_r2', g), np.nan)
            if pd.isna(net_val):
                tex_cols_s_net.append("-")
                csv_cols_s_net.append("-")
            else:
                is_best = abs(net_val - best_r2[('net_f_r2', g)]) < 1e-6
                val_str = f"{net_val:.3f}"
                tex_cols_s_net.append(f"\\textbf{{{val_str}}}" if is_best else val_str)
                csv_cols_s_net.append(val_str)
                
            fric_val = row_data.get(('fric_r2', g), np.nan)
            if pd.isna(fric_val):
                tex_cols_s_fric.append("-")
                csv_cols_s_fric.append("-")
            else:
                is_best = abs(fric_val - best_r2[('fric_r2', g)]) < 1e-6
                val_str = f"{fric_val:.3f}"
                tex_cols_s_fric.append(f"\\textbf{{{val_str}}}" if is_best else val_str)
                csv_cols_s_fric.append(val_str)

        cols_span = str(len(tex_cols_s_net))
        latex_rows_r2_narrow.append("        " + " & ".join(tex_cols_s_net) + " \\\\")
        latex_rows_r2_narrow.append(f"        \\cmidrule(lr){{2-{cols_span}}}")
        latex_rows_r2_narrow.append("        " + " & ".join(tex_cols_s_fric) + " \\\\")
        
        if idx < len(valid_models) - 1:
            latex_rows_r2_narrow.append("        \\midrule")
            
        csv_rows_r2_narrow.extend([csv_cols_s_net, csv_cols_s_fric])

    csv_header_narrow = ["Model", "Law"] + group_order
    pd.DataFrame(csv_rows_r2_narrow, columns=csv_header_narrow).to_csv(os.path.join(output_dir, f"macro_domains_r2_scores{suffix}.csv"), index=False)
    
    r1_s_groups = " & ".join(group_order)
    r1_s = f"        \\textbf{{Model}} & \\textbf{{Law}} & {r1_s_groups} \\\\"

    path = os.path.join(output_dir, f"macro_domains_r2_scores{suffix}.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Physics Law Consistency ($R^2$ Scores) ($\\uparrow$) Across Domains}\n")
        f.write("    \\vspace{2mm}\n")
        f.write("    \\resizebox{\\columnwidth}{!}{\n")
        
        align_s_parts = ["l", "c"] + ["c" for _ in range(len(group_order))]
        align_str = " ".join(align_s_parts)
        
        f.write(f"    \\begin{{tabular}}{{{align_str}}}\n")
        f.write("        \\toprule\n")
        f.write(f"{r1_s}\n")
        f.write("        \\midrule\n")
        f.write("\n".join(latex_rows_r2_narrow) + "\n")
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n    }\n")
        f.write("\\end{table}\n")

    print(f"Generated R2 Consistency tables {suffix} in: {output_dir}")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Processing Simulation Aggregation Tables...")
    
    # We load the data once and pass it to the different table generators
    df_results = load_data(MODELS_TO_COMPARE)
    
    if not df_results.empty:
        # Task 1
        generate_main_tables(df_results, OUTPUT_DIR)
        
        # Task 2
        generate_ablation_tables(df_results, ABLATION_DIR)
        
        # Task 3
        for suffix, domain_groups in ALL_MAPPINGS_R2.items():
            generate_r2_tables(df_results, domain_groups, suffix, PHYSICS_DIR)
            
        print("\nAll aggregation tasks completed successfully.")
    else:
        print("\nNo valid data loaded. Skipping table generation.")