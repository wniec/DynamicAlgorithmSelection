import pandas as pd


def apply_precision_and_bolding(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    aocc_cols = [c for c in df.columns if "AOCC" in str(c)]

    def format_aocc(val):
        if pd.isna(val):
            return "N/A"
        num = float(val)
        formatted = f"{num:.3f}"

        # If this value matches the max, wrap it in our placeholder
        if num == max_val:
            return f"BSTART{formatted}BEND"
        return formatted

    for col in aocc_cols:
        if col not in df.columns:
            continue

        # Find the exact numeric maximum, ignoring NaNs
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        max_val = numeric_col.max()

        df[col] = df[col].apply(format_aocc)

    return df


def get_baseline_mask(df):
    # Check if 'method' is a column or an index level
    if "method" in df.columns:
        return df["method"] == "baseline"
    else:
        return df.index.get_level_values("method") == "baseline"


def save_latex_with_bolding(df: pd.DataFrame, filename: str):
    # 1. Generate the raw LaTeX string (safely escaping underscores)
    latex_str = df.to_latex()

    # 2. Replace the safe placeholders with actual LaTeX bold commands
    latex_str = latex_str.replace("BSTART", r"\textbf{").replace("BEND", "}")

    # 3. Write it to the file
    with open(filename, "w") as f:
        f.write(latex_str)


def format_scenario_df(df: pd.DataFrame, scenario_target: str) -> pd.DataFrame:
    # Move index to a column so we can manipulate it using string methods
    df = df.reset_index()

    # 2. Force the first column (the old index) to be named 'name'
    df.columns.values[0] = "name"

    # Create boolean column for MULTIDIMENSIONAL and then remove the substring
    df["multidimensional"] = df["name"].str.contains("MULTIDIMENSIONAL", na=False)
    df = df[~df["multidimensional"]]
    df = df.drop(columns=["multidimensional"])
    df["name"] = df["name"].str.replace("MULTIDIMENSIONAL", "", regex=False)

    # Remove the scenario substring (e.g., "CV-LOPO" or "CV-LOIO")
    df["name"] = df["name"].str.replace(f"{scenario_target}", "", regex=False)

    # Clean up double/multiple underscores left behind by the deletions
    df["name"] = df["name"].str.replace(r"__+", "_", regex=True)

    # Extract the CDB value (matches "CDB" followed by digits, and optionally a decimal point and more digits)
    df["CDB"] = df["name"].str.extract(r"(CDB\d+(?:\.\d+)?)")[0]

    # delete CDB substring
    df["CDB"] = df["CDB"].fillna("N/A").apply(lambda x: x.replace("CDB", ""))

    # Remove the CDB substring from the name column
    df["name"] = df["name"].str.replace(r"_?CDB\d+(?:\.\d+)?_?", "", regex=True)

    # --- NEW METHOD COLUMN LOGIC ---
    df["method"] = "baseline"  # Default value for baselines/unmatched rows

    # 1. PG -> Exponential-DAS
    mask_pg = df["name"].str.contains("PG", na=False)
    df.loc[mask_pg, "method"] = "Exponential-DAS"
    df["name"] = df["name"].str.replace("PG", "", regex=False)

    # Drop PG with CDB other than 1.0 or 2.1
    mask_cdb_invalid = ~df["CDB"].isin(["1.0", "2.1"])
    rows_to_drop = mask_pg & mask_cdb_invalid
    df = df[~rows_to_drop]

    # 2. RLDAS -> RL-DAS
    mask_rldas = df["name"].str.contains("RLDAS", na=False)
    df.loc[mask_rldas, "method"] = "RL-DAS"
    df["name"] = df["name"].str.replace("RLDAS", "", regex=False)

    # 3. RANDOM_DAS -> Random (Must evaluate BEFORE standard RANDOM)
    mask_random_das = df["name"].str.contains("RANDOM_DAS", na=False)
    df.loc[mask_random_das, "method"] = "Random"
    df["name"] = df["name"].str.replace("RANDOM_DAS", "", regex=False)

    # 4. RANDOM -> Random-Exponential
    mask_random = df["name"].str.contains("RANDOM", na=False)
    df.loc[mask_random, "method"] = "Random-Exponential"
    df["name"] = df["name"].str.replace("RANDOM", "", regex=False)

    rows_to_drop = mask_random & mask_cdb_invalid
    df = df[~rows_to_drop]

    # Strip any trailing, leading, or double underscores left by deletions
    df["name"] = df["name"].str.replace(r"__+", "_", regex=True).str.strip("_")
    df["name"] = (
        df["name"]
        .str.replace(r"BASELINES_|DAS_CV_|\.html|DAS_", "", regex=True)
        .str.strip("_")
    )
    df["name"] = df["name"].str.replace(r"baselines", "", regex=True).str.strip("_")
    df["name"] = df["name"].str.replace(r"_", " ", regex=True)
    # -------------------------------

    # Set the new composite primary key
    # (You can add 'method' to this index list if you want it to be part of the MultiIndex)
    df = df.groupby(["name", "CDB", "method"]).first()
    df.index.names = [
        "algorithm/portfolio" if x == "name" else x for x in df.index.names
    ]

    # 2. Reset the index so we can manipulate the columns for sorting
    df = df.reset_index()

    # 3. Create a temporary column for the length of the algorithm name
    df["name_len"] = df["algorithm/portfolio"].str.len()

    # 4. Perform the multi-level sort:
    #    - Length of name (Ascending)
    #    - Value of name (Alphabetical)
    #    - CDB value
    #    - Multidimensional (Boolean)
    #    - Method
    df = df.sort_values(
        by=["name_len", "algorithm/portfolio", "CDB", "method"],
        ascending=[True, True, True, True],
    )

    # 5. Drop the helper column and set the index back if you want it as a MultiIndex again
    df = df.drop(columns=["name_len"])
    df = df.set_index(["algorithm/portfolio", "CDB", "method"])
    return df


def get_output_latex(loio_tables, lopo_tables):
    # Combine all collected tables and save to a single LaTeX file
    if loio_tables and lopo_tables:
        print("\n" + "=" * 60)
        print("JOINING AND SAVING TABLES")
        print("=" * 60)

        # Concatenate along columns; Pandas aligns the indices automatically
        loio_df = pd.concat(loio_tables, axis=1)
        lopo_df = pd.concat(lopo_tables, axis=1)

        print("\n" + "=" * 60)
        print("POST-PROCESSING: SPLITTING & CLEANING (LOPO / LOIO / REWARD)")
        print("=" * 60)

        # 1. Extract REWARD dataset into its own dataframe
        # Uses regex to match 'REWARD' followed by a digit
        df_reward = lopo_df[
            lopo_df.index.str.contains(r"REWARD\d", regex=True, na=False)
        ].copy()

        # Remove REWARD rows from the main dataframe to keep datasets mutually exclusive
        lopo_df = lopo_df[
            ~lopo_df.index.str.contains(r"REWARD\d", regex=True, na=False)
        ].copy()

        # 3. Process both datasets
        df_lopo_clean = format_scenario_df(lopo_df, "CV-LOPO")
        df_loio_clean = format_scenario_df(loio_df, "CV-LOIO")

        # 4. Save the separated and cleaned dataframes
        output_lopo = "COMBINED_METRICS_LOPO.tex"
        output_loio = "COMBINED_METRICS_LOIO.tex"
        output_reward = "reward.tex"
        output_baselines = "baselines.tex"

        lopo_mask = get_baseline_mask(df_lopo_clean)
        df_baselines = df_lopo_clean[lopo_mask].copy()
        df_lopo_clean = df_lopo_clean[~lopo_mask]

        # For LOIO: Just delete the rows (keep only where mask is False)
        loio_mask = get_baseline_mask(df_loio_clean)
        df_loio_clean = df_loio_clean[~loio_mask]

        # Split baselines: single-algorithm runs vs. population-based competitors
        _SINGLE_ALGO = {"G3PCX", "LMCMAES", "SPSO"}
        algo_names = df_baselines.index.get_level_values("algorithm/portfolio")
        single_algo_mask = algo_names.isin(_SINGLE_ALGO)
        df_single_algo = df_baselines[single_algo_mask].copy()
        df_baselines = df_baselines[~single_algo_mask].copy()

        df_lopo_clean.columns = df_lopo_clean.columns.str.replace("_", " ")
        df_loio_clean.columns = df_loio_clean.columns.str.replace("_", " ")

        df_lopo_clean = apply_precision_and_bolding(df_lopo_clean)
        df_loio_clean = apply_precision_and_bolding(df_loio_clean)
        if not df_baselines.empty:
            df_baselines.columns = df_baselines.columns.str.replace("_", " ")
        if not df_single_algo.empty:
            df_single_algo.columns = df_single_algo.columns.str.replace("_", " ")

        save_latex_with_bolding(df_lopo_clean, output_lopo)
        save_latex_with_bolding(df_loio_clean, output_loio)
        if not df_baselines.empty:
            save_latex_with_bolding(df_baselines, output_baselines)
            print(
                f"Saved BASELINES dataset to {output_baselines} (Shape: {df_baselines.shape})"
            )
        if not df_single_algo.empty:
            save_latex_with_bolding(df_single_algo, "single_algo_baselines.tex")
            print(
                f"Saved single-algorithm baselines to single_algo_baselines.tex"
                f" (Shape: {df_single_algo.shape})"
            )

        print(
            f"Saved cleaned LOPO dataset to {output_lopo} (Shape: {df_lopo_clean.shape})"
        )
        print(
            f"Saved cleaned LOIO dataset to {output_loio} (Shape: {df_loio_clean.shape})"
        )

        # Save REWARD dataset if it contains any rows
        if not df_reward.empty:
            df_reward.to_latex(output_reward)
            print(f"Saved REWARD dataset to {output_reward} (Shape: {df_reward.shape})")
        else:
            print("No REWARD rows found. 'reward.tex' was not created.")
