import pandas


def main():
    """Entry point."""

    # Load US gross output by industry
    us_gross_output_1997_2022 = './data/US Gross Output by Industry 1997-2022.csv'
    us_gross_output_1960_1997 = './data/US Gross Output by Industry 1960-1997.csv'
    table = None
    for table_path in [us_gross_output_1997_2022, us_gross_output_1960_1997]:
        local_table = pandas.read_csv(
            table_path, skiprows=[0, 1, 2])
        local_table.dropna(subset=['1997'], inplace=True)
        print(local_table)
        if table is None:
            table = local_table
        else:
            table = pandas.merge(
                table, local_table, left_index=True, right_index=True,
                suffixes=('', '_dup'))
    table = table.rename(columns={'Unnamed: 1': 'Industry'})
    table = table.filter(regex='^(?!.*_dup$)')
    table = table.set_index(table.columns[1])
    table = table.drop('Line', axis=1)

    co2_emissions_table = pandas.read_csv(
        './data/annual-co2-emissions-per-country.csv')
    co2_emissions_table = co2_emissions_table[
        co2_emissions_table['Year'].astype(str).isin(set(table.columns))]
    co2_emissions_table.dropna(subset=['Code'], inplace=True)
    co2_emissions_table = co2_emissions_table.pivot(
        index='Entity', columns='Year', values=co2_emissions_table.columns[3])
    print(co2_emissions_table)
    print(table)

    years = list(sorted(table.columns))
    print(years)



if __name__ == '__main__':
    main()
