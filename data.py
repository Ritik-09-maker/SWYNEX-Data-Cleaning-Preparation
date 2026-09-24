# ============================================================
# STUDENTS PERFORMANCE DATASET
# COMPLETE DATA CLEANING & PREPARATION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "StudentsPerformance.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. CREATE A COPY
# ============================================================

# Keep original dataset safe
original_df = df.copy()


# ============================================================
# 4. INITIAL DATA INSPECTION
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
display(df.head())

print("\n========== LAST 5 ROWS ==========")
display(df.tail())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATA INFO ==========")
df.info()


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\n========== NUMERICAL SUMMARY ==========")
display(df.describe())

print("\n========== CATEGORICAL SUMMARY ==========")
display(df.describe(include="object"))


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_count = df.isnull().sum()

missing_percentage = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

display(missing_table)


# ============================================================
# 7. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n========== CLEAN COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# 8. REMOVE COMPLETELY EMPTY ROWS/COLUMNS
# ============================================================

# Remove rows where every value is missing
df.dropna(how="all", inplace=True)

# Remove columns where every value is missing
df.dropna(axis=1, how="all", inplace=True)

print("\nShape after removing completely empty rows/columns:")
print(df.shape)


# ============================================================
# 9. REMOVE DUPLICATE ROWS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 10. STRIP EXTRA SPACES FROM TEXT COLUMNS
# ============================================================

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].str.strip()

print("\nExtra spaces removed from text columns.")


# ============================================================
# 11. STANDARDIZE TEXT FORMAT
# ============================================================

# Convert text values to lowercase temporarily
# This helps identify inconsistent categories.

for column in text_columns:
    df[column] = df[column].str.lower()

print("\nText values standardized to lowercase.")


# ============================================================
# 12. CHECK UNIQUE CATEGORICAL VALUES
# ============================================================

print("\n========== UNIQUE CATEGORICAL VALUES ==========")

for column in text_columns:
    print(f"\n{column}:")
    print(df[column].unique())


# ============================================================
# 13. STANDARDIZE KNOWN CATEGORICAL VALUES
# ============================================================

# Gender
if "gender" in df.columns:
    df["gender"] = df["gender"].replace({
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female"
    })


# Race / Ethnicity
if "race/ethnicity" in df.columns:
    df["race/ethnicity"] = df["race/ethnicity"].str.strip()


# Lunch
if "lunch" in df.columns:
    df["lunch"] = df["lunch"].replace({
        "standard": "standard",
        "free/reduced": "free/reduced"
    })


# Test preparation
if "test_preparation_course" in df.columns:
    df["test_preparation_course"] = df[
        "test_preparation_course"
    ].replace({
        "none": "none",
        "completed": "completed"
    })


print("\nCategorical values standardized.")


# ============================================================
# 14. CONVERT NUMERICAL COLUMNS TO NUMERIC
# ============================================================

score_columns = [
    "math_score",
    "reading_score",
    "writing_score"
]

for column in score_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print("\nNumerical columns converted successfully.")


# ============================================================
# 15. HANDLE MISSING VALUES
# ============================================================

# Numerical columns → Median
for column in score_columns:
    if column in df.columns:

        if df[column].isnull().sum() > 0:
            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )


# Categorical columns → Mode
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if len(mode_value) > 0:
            df[column] = df[column].fillna(
                mode_value[0]
            )


print("\nMissing values handled.")


# ============================================================
# 16. CHECK INVALID SCORE VALUES
# ============================================================

print("\n========== INVALID SCORE VALUES ==========")

for column in score_columns:

    if column in df.columns:

        invalid = df[
            (df[column] < 0) |
            (df[column] > 100)
        ]

        print(
            f"{column}: {len(invalid)} invalid values"
        )


# ============================================================
# 17. HANDLE INVALID SCORE VALUES
# ============================================================

for column in score_columns:

    if column in df.columns:

        # Replace invalid values with NaN
        df.loc[
            (df[column] < 0) |
            (df[column] > 100),
            column
        ] = np.nan

        # Fill with median
        df[column] = df[column].fillna(
            df[column].median()
        )


print("\nInvalid score values handled.")


# ============================================================
# 18. OUTLIER DETECTION USING IQR
# ============================================================

def detect_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


print("\n========== OUTLIER ANALYSIS ==========")

for column in score_columns:

    outliers, lower, upper = detect_outliers_iqr(
        df,
        column
    )

    print(f"\n{column}")
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Outliers:", len(outliers))


# ============================================================
# 19. OUTLIER TREATMENT USING IQR CLIPPING
# ============================================================

# Instead of deleting students, we cap extreme values.
# This preserves the records.

for column in score_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_bound,
        upper_bound
    )


print("\nOutliers treated using IQR clipping.")


# ============================================================
# 20. FEATURE ENGINEERING
# ============================================================

# Average score
df["average_score"] = df[
    score_columns
].mean(axis=1)


# Total score
df["total_score"] = df[
    score_columns
].sum(axis=1)


# ============================================================
# 21. CREATE PERFORMANCE CATEGORY
# ============================================================

def performance_category(score):

    if score >= 80:
        return "excellent"

    elif score >= 60:
        return "good"

    elif score >= 40:
        return "average"

    else:
        return "poor"


df["performance"] = df[
    "average_score"
].apply(performance_category)


# ============================================================
# 22. CREATE PASS/FAIL COLUMN
# ============================================================

# Student is considered pass if all three scores >= 40

df["pass_status"] = np.where(
    (df["math_score"] >= 40) &
    (df["reading_score"] >= 40) &
    (df["writing_score"] >= 40),
    "pass",
    "fail"
)


# ============================================================
# 23. CHECK DATA AFTER CLEANING
# ============================================================

print("\n========== DATA AFTER CLEANING ==========")

display(df.head())

print("\nShape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 24. VISUALIZE SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

df[score_columns].boxplot()

plt.title("Student Score Distribution")
plt.ylabel("Score")

plt.show()


# ============================================================
# 25. CHECK FINAL STATISTICS
# ============================================================

print("\n========== FINAL STATISTICS ==========")

display(
    df[
        [
            "math_score",
            "reading_score",
            "writing_score",
            "average_score",
            "total_score"
        ]
    ].describe()
)


# ============================================================
# 26. OPTIONAL: ENCODE CATEGORICAL VARIABLES
# ============================================================

# Create a separate ML-ready dataset
# so that the original cleaned dataset remains readable.

df_ml = pd.get_dummies(
    df,
    columns=df.select_dtypes(
        include="object"
    ).columns,
    drop_first=False
)

print("\n========== ML-READY DATASET ==========")

print("Shape:", df_ml.shape)

display(df_ml.head())


# ============================================================
# 27. SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

scaling_columns = [
    "math_score",
    "reading_score",
    "writing_score",
    "average_score",
    "total_score"
]

df_ml[scaling_columns] = scaler.fit_transform(
    df_ml[scaling_columns]
)

print("\nNumerical features standardized.")


# ============================================================
# 28. FINAL VALIDATION
# ============================================================

print("\n==========================================")
print("        FINAL DATA VALIDATION")
print("==========================================")

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)

print(
    "Total duplicate rows:",
    df.duplicated().sum()
)

print(
    "Total invalid scores:"
)

for column in score_columns:

    invalid_count = (
        (df[column] < 0) |
        (df[column] > 100)
    ).sum()

    print(
        f"{column}: {invalid_count}"
    )


# ============================================================
# 29. SAVE CLEANED DATASET
# ============================================================

cleaned_file = "StudentsPerformance_Cleaned.csv"

df.to_csv(
    cleaned_file,
    index=False
)

print(
    f"\nCleaned dataset saved as: {cleaned_file}"
)


# ============================================================
# 30. SAVE ML-READY DATASET
# ============================================================

ml_file = "StudentsPerformance_ML_Ready.csv"

df_ml.to_csv(
    ml_file,
    index=False
)

print(
    f"ML-ready dataset saved as: {ml_file}"
)


# ============================================================
# 31. FINAL PREVIEW
# ============================================================

print("\n========== FINAL CLEANED DATASET ==========")

display(df.head(10))

print("\n========== FINAL COLUMNS ==========")

print(df.columns.tolist())