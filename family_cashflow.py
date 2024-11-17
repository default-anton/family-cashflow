import csv
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

class UnsupportedBankError(Exception):
    pass

class TransactionProcessor:
    def _detect_owner(self, file_path):
        """Detect the owner based on filename."""
        filename = file_path.lower()

        if 'anton' in filename:
            return 'Anton'
        elif 'anna' in filename:
            return 'Anna'

        raise ValueError(f"Could not detect owner (Anton/Anna) from filename: {file_path}")

    def _detect_bank_format(self, file_path):
        """Detect the bank format based on CSV structure."""
        # Try reading as RBC first (with headers)
        try:
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
        except:
            pass

        # Try reading first row of CIBC (no headers)
        try:
            df_first_row = pd.read_csv(
                file_path,
                nrows=1,
                header=None,
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True,
            )

            # CIBC files have 5 columns: Date, Description, Debit, Credit, Card Number
            if (len(df_first_row.columns) == 5 and
                pd.to_datetime(df_first_row.iloc[0, 0], format='%Y-%m-%d') and
                '*' in str(df_first_row.iloc[0, 4])):  # Check for asterisks in card number
                return "CIBC"
        except:
            pass

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

    def _read_and_process_cibc(self, file_path):
        df = pd.read_csv(
            file_path,
            header=None,
            names=['Date', 'Description', 'Debit', 'Credit', 'Card'],
            index_col=False,
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )

        # Convert Debit/Credit to Amount (Debit is negative, Credit is positive)
        df['Amount'] = df['Credit'].fillna(0) - df['Debit'].fillna(0)
        df['Description'] = df['Description'].str.strip()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        result = df[['Date', 'Description', 'Amount']].copy()
        result = self._filter_internal_transfers(result)

        return result

    def read_and_process(self, file_path):
        """Main interface to read and process bank transaction files."""
        bank_format = self._detect_bank_format(file_path)
        owner = self._detect_owner(file_path)

        df = None
        if bank_format == "RBC":
            df = self._read_and_process_rbc(file_path)
        elif bank_format == "CIBC":
            df = self._read_and_process_cibc(file_path)
        else:
            raise UnsupportedBankError("This file format is not supported yet.")

        df['Owner'] = owner

        return df.sort_values('Date', ascending=False)

def main():
    parser = argparse.ArgumentParser(description='Process bank transactions')
    parser.add_argument('file_paths', nargs='+', help='Paths to the CSV files')
    parser.add_argument('--month', help='Month to filter transactions (YYYY-MM format)')
    args = parser.parse_args()

    processor = TransactionProcessor()
    all_transactions = []

    for file_path in args.file_paths:
        try:
            transactions = processor.read_and_process(file_path)
            all_transactions.append(transactions)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not all_transactions:
        print("No transactions were successfully processed")
        return

    transactions = pd.concat(all_transactions, ignore_index=True)

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

    transactions = transactions[['Date', 'Owner', 'Description', 'Amount']]
    transactions = transactions.sort_values('Date', ascending=False)

    owners = '_'.join(sorted([owner.lower() for owner in transactions['Owner'].unique()]))
    output_filename = f"{owners}_{'all' if not args.month else args.month}_processed_family_cashflow.csv"
    output_path = Path(args.file_paths[0]).parent.joinpath(output_filename)
    transactions.to_csv(output_path, index=False)

    print(f"Transactions saved to {output_path}")

    if sys.platform == "darwin":
        try:
            subprocess.run(["open", output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error opening file: {e}")

if __name__ == "__main__":
    main()
