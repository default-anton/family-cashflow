import csv
import argparse
import subprocess
import sys
import pandas as pd

class UnsupportedBankError(Exception):
    pass

class TransactionProcessor:
    def __init__(self, owner):
        self.owner = owner

    def _detect_bank_format(self, file_path):
        """Detect the bank format based on CSV headers."""
        df_header = pd.read_csv(
            file_path,
            nrows=0,
            index_col=False,
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )

        rbc_columns = {"Description 1", "Description 2", "Transaction Date", "CAD$"}
        if rbc_columns.issubset(df_header.columns):
            return "RBC"

        return None

    def _filter_internal_transfers(self, df):
        """Remove matching internal transfer transactions."""
        indices_to_remove = set()

        for date in df['Date'].unique():
            date_df = df[df['Date'] == date]
            outgoing_mask = date_df['Description'].str.contains('WWW TRF DDA', case=False, na=False)
            incoming_mask = date_df['Description'].str.contains('Transfer WWW TRANSFER', case=False, na=False)
            outgoing = date_df[outgoing_mask]
            incoming = date_df[incoming_mask]

            # Match transfers with same absolute amount
            for _, out_row in outgoing.iterrows():
                amount = abs(out_row['Amount'])
                matching_in = incoming[incoming['Amount'] == amount]

                if not matching_in.empty:
                    indices_to_remove.add(out_row.name)
                    indices_to_remove.add(matching_in.iloc[0].name)

        return df[~df.index.isin(indices_to_remove)]

    def _read_and_process_rbc(self, file_path):
        df = pd.read_csv(
            file_path,
            index_col=False,
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )
        df['Description'] = df['Description 1'].fillna('') + ' ' + df['Description 2'].fillna('')
        df['Description'] = df['Description'].str.strip()
        df['Date'] = pd.to_datetime(df['Transaction Date']).dt.strftime('%Y-%m-%d')
        df['Amount'] = df['CAD$']
        result = df[['Date', 'Description', 'Amount']].copy()
        result = self._filter_internal_transfers(result)

        return result

    def read_and_process(self, file_path):
        """Main interface to read and process bank transaction files."""
        bank_format = self._detect_bank_format(file_path)

        df = None
        if bank_format == "RBC":
            df = self._read_and_process_rbc(file_path)

        if df is None:
            raise UnsupportedBankError("This file format is not supported yet.")

        return df.sort_values('Date', ascending=False)

def main():
    parser = argparse.ArgumentParser(description='Process bank transactions')
    parser.add_argument('file_path', help='Path to the CSV file')
    parser.add_argument('--owner', default='Anton', choices=['Anton', 'Anna'], help='Owner of the transactions (Anton or Anna)')
    parser.add_argument('--month', help='Month to filter transactions (YYYY-MM format)')
    args = parser.parse_args()

    processor = TransactionProcessor(args.owner)
    transactions = processor.read_and_process(args.file_path)

    if args.month:
        try:
            filter_date = pd.to_datetime(args.month + '-01')
            transactions['DateObj'] = pd.to_datetime(transactions['Date'])
            mask = (transactions['DateObj'].dt.year == filter_date.year) & \
                   (transactions['DateObj'].dt.month == filter_date.month)
            transactions = transactions[mask]
            transactions.drop('DateObj', axis=1, inplace=True)

            if transactions.empty:
                print(f"No transactions found for {args.month}")
                return
        except ValueError:
            print("Error: Month must be in YYYY-MM format (e.g., 2024-01)")
            return

    transactions['Owner'] = args.owner
    transactions = transactions[['Date', 'Owner', 'Description', 'Amount']]
    
    # Generate output filename
    base_path = args.file_path.rsplit('.', 1)[0]
    output_path = f"{base_path}_{'all' if not args.month else args.month}_processed.csv"
    transactions.to_csv(output_path, index=False)

    print(f"Transactions saved to {output_path}")
    
    if sys.platform == "darwin":
        try:
            subprocess.run(["open", output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error opening file: {e}")

if __name__ == "__main__":
    main()
