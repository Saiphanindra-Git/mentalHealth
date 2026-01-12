import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import RAW_DATA_FILE, PROCESSED_DATA_DIR, RESULTS_DIR, RANDOM_SEED
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_raw_data():
    """Load the raw dataset"""
    print("="*60)
    print("STEP 1: LOADING RAW DATA")
    print("="*60)
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.columns.tolist()}")
    return df

def initial_inspection(df):
    """Inspect raw data quality"""
    print("\n" + "="*60)
    print("STEP 2: INITIAL DATA INSPECTION")
    print("="*60)
    
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # Show sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    return df

def clean_dataset(df):
    """Clean and structure the dataset"""
    print("\n" + "="*60)
    print("STEP 3: DATA CLEANING")
    print("="*60)
    
    original_count = len(df)
    
    # 1. Drop the index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✓ Dropped unnecessary index column")
    
    # 2. Rename columns for clarity
    df.columns = ['text', 'label']
    print("✓ Renamed columns to 'text' and 'label'")
    
    # 3. Handle missing values in text
    missing_text = df['text'].isnull().sum()
    if missing_text > 0:
        print(f"✓ Removing {missing_text} rows with missing text")
        df = df.dropna(subset=['text'])
    
    # 4. Remove empty or very short texts
    df['text'] = df['text'].astype(str)
    df['text_length'] = df['text'].str.len()
    short_texts = len(df[df['text_length'] < 3])
    df = df[df['text_length'] >= 3]
    if short_texts > 0:
        print(f"✓ Removed {short_texts} rows with text length < 3 characters")
    
    # 5. Remove duplicates
    duplicates = df.duplicated(subset=['text']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f"✓ Removed {duplicates} duplicate rows")
    
    # 6. Clean text data
    print("✓ Cleaning text data...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    # 7. Remove rows where cleaned text is too short
    df['cleaned_length'] = df['text_cleaned'].str.len()
    very_short = len(df[df['cleaned_length'] < 3])
    df = df[df['cleaned_length'] >= 3]
    if very_short > 0:
        print(f"✓ Removed {very_short} rows with cleaned text < 3 characters")
    
    # 8. Reset index
    df = df.reset_index(drop=True)
    
    print(f"\n📊 Cleaning Summary:")
    print(f"   Original rows: {original_count}")
    print(f"   Final rows: {len(df)}")
    print(f"   Rows removed: {original_count - len(df)} ({((original_count - len(df))/original_count*100):.2f}%)")
    
    return df

def clean_text(text):
    """Clean individual text entries"""
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Reddit-style links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,!?\'-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def analyze_labels(df):
    """Analyze label distribution"""
    print("\n" + "="*60)
    print("STEP 4: LABEL ANALYSIS")
    print("="*60)
    
    print(f"\nUnique labels: {df['label'].unique()}")
    print(f"Number of classes: {df['label'].nunique()}")
    
    print(f"\n📊 Label Distribution:")
    label_counts = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({
        'Count': label_counts,
        'Percentage': label_percentages.round(2)
    })
    print(summary_df)
    
    # Check class imbalance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n⚠️  Class imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 5:
        print("    (High imbalance - may need class weighting during training)")
    
    # Visualize distribution
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)
    plt.title('Mental Health Label Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Mental Health Status', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(label_counts):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'plots', 'label_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()
    
    return df

def analyze_text_stats(df):
    """Analyze text characteristics"""
    print("\n" + "="*60)
    print("STEP 5: TEXT STATISTICS")
    print("="*60)
    
    # Calculate statistics
    df['word_count'] = df['text_cleaned'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text_cleaned'].apply(lambda x: len(str(x)))
    
    print(f"\n📊 Text Length Statistics (Cleaned Text):")
    print(f"\nCharacter count:")
    print(df['char_count'].describe().round(2))
    print(f"\nWord count:")
    print(df['word_count'].describe().round(2))
    
    # Statistics by label
    print(f"\n📊 Average Text Length by Label:")
    label_stats = df.groupby('label').agg({
        'char_count': 'mean',
        'word_count': 'mean'
    }).round(2)
    label_stats.columns = ['Avg Chars', 'Avg Words']
    print(label_stats)
    
    # Show samples from each class
    print(f"\n📝 Sample Texts from Each Class:")
    for label in df['label'].unique():
        sample = df[df['label'] == label].iloc[0]['text_cleaned'][:150]
        print(f"\n{label}:")
        print(f"  {sample}...")
    
    return df

def split_and_save_data(df):
    """Split data and save to processed folder"""
    print("\n" + "="*60)
    print("STEP 6: SPLITTING AND SAVING DATA")
    print("="*60)
    
    # Keep only necessary columns
    df_final = df[['text_cleaned', 'label']].copy()
    df_final.columns = ['text', 'label']
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df_final, 
        test_size=0.30, 
        random_state=RANDOM_SEED,
        stratify=df_final['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['label']
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training set:   {len(train_df):,} samples ({len(train_df)/len(df_final)*100:.1f}%)")
    print(f"   Validation set: {len(val_df):,} samples ({len(val_df)/len(df_final)*100:.1f}%)")
    print(f"   Test set:       {len(test_df):,} samples ({len(test_df)/len(df_final)*100:.1f}%)")
    
    # Show label distribution in each split
    print(f"\n📊 Label Distribution in Splits:")
    print(f"\nTraining:")
    print(train_df['label'].value_counts())
    print(f"\nValidation:")
    print(val_df['label'].value_counts())
    print(f"\nTest:")
    print(test_df['label'].value_counts())
    
    # Create directory if not exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save to CSV
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Data saved successfully:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    return train_df, val_df, test_df

def create_summary_report(df, train_df, val_df, test_df):
    """Create final summary report"""
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    
    print(f"\n✅ Dataset Successfully Cleaned and Processed!")
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Number of classes: {df['label'].nunique()}")
    print(f"   Classes: {', '.join(df['label'].unique())}")
    print(f"   Average text length: {df['char_count'].mean():.0f} characters")
    print(f"   Average word count: {df['word_count'].mean():.0f} words")
    
    print(f"\n✅ Data Split Completed:")
    print(f"   Train:      {len(train_df):,} samples")
    print(f"   Validation: {len(val_df):,} samples")
    print(f"   Test:       {len(test_df):,} samples")
    
    print(f"\n✅ Files Ready for Model Training:")
    print(f"   ✓ data/processed/train.csv")
    print(f"   ✓ data/processed/val.csv")
    print(f"   ✓ data/processed/test.csv")
    
    print(f"\n🚀 Next Step:")
    print(f"   Run: python src\\train.py")
    print(f"   (to start BERT model training)")

def main():
    """Main execution pipeline"""
    print("\n" + "="*60)
    print("MENTAL HEALTH DATA CLEANING & PREPROCESSING")
    print("="*60)
    
    # Create results directory
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Execute pipeline
    df = load_raw_data()
    df = initial_inspection(df)
    df = clean_dataset(df)
    df = analyze_labels(df)
    df = analyze_text_stats(df)
    train_df, val_df, test_df = split_and_save_data(df)
    create_summary_report(df, train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
