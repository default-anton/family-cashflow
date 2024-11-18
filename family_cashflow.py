import csv
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import timedelta

import requests
import pandas as pd
from numbers_parser import Document

class UnsupportedBankError(Exception):
    pass

class TransactionProcessor:
    def __init__(self, owners):
        """Initialize with a list of owners."""
        self.owners = owners
        self.categories_df = None

    def load_categories(self, numbers_file):
        """Load transaction categories from a Numbers file."""
        doc = Document(numbers_file)
        sheets = doc.sheets
        tables = sheets[0].tables
        data = tables[0].rows(values_only=True)
        self.categories_df = pd.DataFrame(data[1:], columns=data[0])
        self.categories_df['Description'] = self.categories_df['Description'].str.lower()

    def categorize_transaction(self, description):
        """Categorize a transaction based on its description."""
        if self.categories_df is None:
            return ''

        description = description.lower()
        for _, row in self.categories_df.iterrows():
            if row['Description'] in description:
                return row['Category']

        return ''

    def _detect_owner(self, file_path):
        """Detect the owner based on filename."""
        filename = file_path.lower()

        for owner in self.owners:
            if owner.lower() in filename:
                return owner

        raise ValueError(f"Could not detect owner ({', '.join(self.owners)}) from filename: {file_path}")

    def _detect_bank_format(self, file_path):
        """Detect the bank format based on CSV structure."""
        # Try reading as RBC or Wise (with headers)
        df_header = pd.read_csv(
            file_path,
            nrows=0,
            index_col=False,
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )

        wise_columns = {"ID", "Status", "Direction", "Source amount (after fees)", "Target amount (after fees)"}
        if wise_columns.issubset(df_header.columns):
            return "WISE"

        rbc_columns = {"Description 1", "Description 2", "Transaction Date", "CAD$"}
        if rbc_columns.issubset(df_header.columns):
            return "RBC"

        # Try reading first row of CIBC (no headers)
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

        raise UnsupportedBankError(f"Could not detect bank format for file: {file_path}")

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
        df['Currency'] = 'CAD'
        result = df[['Date', 'Description', 'Currency', 'Amount']].copy()

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
        df['Currency'] = 'CAD'
        result = df[['Date', 'Description', 'Currency', 'Amount']].copy()

        return result

    def _get_exchange_rates(self, start_date, end_date, currencies):
        """Fetch exchange rates from Bank of Canada for the given date range and currencies."""
        # Remove CAD and create FX codes for Bank of Canada API
        currencies = set(currencies) - {'CAD'}
        if not currencies:
            return {}

        fx_codes = [f"FX{curr}CAD" for curr in currencies]
        url = f"https://www.bankofcanada.ca/valet/observations/{','.join(fx_codes)}/json"
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'order_dir': 'asc'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        assert 'observations' in data, "Invalid response from Bank of Canada API"
        assert len(data['observations']) > 0, "No exchange rate data available for the specified period"

        rates = {}
        for obs in data['observations']:
            date = obs['d']
            rates[date] = {}
            for curr in currencies:
                fx_code = f'FX{curr}CAD'
                assert fx_code in obs, f"Missing exchange rate for {curr} on {date}"
                rates[date][curr] = float(obs[fx_code]['v'])

        return rates

    def _find_closest_previous_date(self, date, exchange_rates):
        """Find the closest previous date in exchange_rates."""
        assert exchange_rates, "Exchange rates dictionary cannot be empty"
        
        previous_dates = [d for d in exchange_rates if d <= date]
        assert previous_dates, f"No exchange rates found for or before date {date}"
        
        return max(previous_dates)

    def _read_and_process_wise(self, file_path):
        """Process Wise transaction history CSV file."""
        df = pd.read_csv(
            file_path,
            index_col=False,
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True
        )

        df['Date'] = pd.to_datetime(df['Finished on']).dt.strftime('%Y-%m-%d')
        dates = pd.to_datetime(df['Date'].unique())
        start_date = min(dates) - timedelta(days=7)
        end_date = max(dates) + timedelta(days=1)

        currencies = set(df['Source currency'].unique()) | set(df['Target currency'].unique())
        exchange_rates = self._get_exchange_rates(start_date, end_date, currencies)

        def convert_amount(row):
            target_currency = row['Target currency']
            target_amount = row['Target amount (after fees)']
            if target_currency == 'CAD':
                return target_amount

            closest_date = self._find_closest_previous_date(row['Date'], exchange_rates)

            assert closest_date is not None and target_currency in exchange_rates[closest_date], \
              f"CAD exchange rate not found for {target_currency} on {row['Date']}. Closest date: {closest_date}"

            rate = exchange_rates[closest_date][target_currency]
            return target_amount * rate

        df['Amount'] = df.apply(convert_amount, axis=1)
        # Make outgoing transactions negative
        df.loc[df['Direction'] == 'OUT', 'Amount'] *= -1
        df['Description'] = df['Source name'] + ' → ' + df['Target name']
        df['Currency'] = 'CAD'

        result = df[['Date', 'Description', 'Currency', 'Amount']].copy()
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
        elif bank_format == "WISE":
            df = self._read_and_process_wise(file_path)
        else:
            raise UnsupportedBankError("This file format is not supported yet.")

        df['Owner'] = owner
        df['Institution'] = bank_format
        df['Category'] = df['Description'].apply(self.categorize_transaction)

        return df.sort_values('Date', ascending=True)

def main():
    parser = argparse.ArgumentParser(description='Process bank transactions')
    parser.add_argument('file_paths', nargs='+', help='Paths to the CSV files')
    parser.add_argument('--month', help='Month to filter transactions (YYYY-MM format)')
    parser.add_argument('--owners', nargs='+', required=True, help='List of owners to detect in filenames')
    parser.add_argument('--categories', help='Path to the transaction categories Numbers file')
    args = parser.parse_args()

    processor = TransactionProcessor(args.owners)
    if args.categories:
        processor.load_categories(args.categories)
    all_transactions = []

    for file_path in args.file_paths:
        transactions = processor.read_and_process(file_path)
        all_transactions.append(transactions)

    assert all_transactions, "No transactions were processed"

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

    transactions = transactions[['Date', 'Owner', 'Institution', 'Description', 'Amount', 'Currency', 'Category']]
    transactions = transactions.sort_values('Date', ascending=True)

    output_filename = f"{'all' if not args.month else args.month}_family_cashflow.csv"
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
