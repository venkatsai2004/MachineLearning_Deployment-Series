def create_features(df):
    # Create new features based on existing data
    df['dti'] = df['loan_amount'] / df['income']
    
    bins = [0, 3000, 6000, 10000, float('inf')]
    labels = ['Low', 'Medium', 'High', 'Very High']
    df['income_bracket'] = pd.cut(df['income'], bins=bins, labels=labels)

    bins = [0, 20000, 50000, 100000, float('inf')]
    labels = ['Small', 'Medium', 'Large', 'Very Large']
    df['loan_bracket'] = pd.cut(df['loan_amount'], bins=bins, labels=labels)

    return df