import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
import os
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

class ContractsDataHandler(object):
    def __init__(self, **kwargs):

        # The path to the excel file being processed
        self._path = kwargs['path'] if 'path' in kwargs else None

        # A list of sheets which we want to process
        self._sheets_to_read = kwargs['sheets_to_read'] if 'sheets_to_read' in kwargs else None

        # The path to the folder in which we want to save some results
        self._folder_path = kwargs['folder_path'] if 'folder_path' in kwargs else None

        # Create this directory if it does not exist
        os.makedirs(self._folder_path, exist_ok=True)

        # A dictionary in which we can override the rows to keep based on the number of previous months
        self._num_months_dict = kwargs['num_months_dict'] if 'num_months_dict' in kwargs else None

        # A dictionary containing a dataframe per contract
        self.dictionary_of_contracts = defaultdict(dict) # default value of defaultdict is dict


    def read_excel(self):
        """

        Load the excel file
        :return:
        """

        xl_file = pd.ExcelFile(self._path)

        #  Get a list of the sheet names in this excel file
        sheet_names = xl_file.sheet_names

        # including only the specified sheets
        for sheet_name in sheet_names:

            if sheet_name not in self._sheets_to_read:
                continue

            self.read_sheet(xl_file, sheet_name)
        


    def create_contract_col_mapper(self, df):

        # Get the names of the contracts asnd store as list
        contract_row_values = df.iloc[0].to_list()

        # Store the col_ids for each contract (finding the index of all cols that have a contract)
        # works by checking if it's a float or not as a nan is not a float
        contract_col_ids = [(col_id, val_) for (col_id, val_) in enumerate(contract_row_values) if
                            not isinstance(val_, float)]

        # Create a map between the name of each column, and the col_ids it corresponds
        contract_col_mapper = dict()

        # Using zip as it pairs each entry with the last and automatically terminates at the last pair
        # We are matching the relevant columns to their contract
        # dealing w this in contract_col_mapper which is a dict where the keys are the
        # contract name and the vals are lists with the columns to be populated
        # contract_1 and 2 are tuples of col_id and contract
        for (contract_1, contract_2) in zip(contract_col_ids, contract_col_ids[1:]):
            contract_to_consider = contract_1[1]
            col_ids = list(range(contract_1[0], contract_2[0]))
            contract_col_mapper[contract_to_consider] = col_ids

        return contract_col_mapper

    def read_sheet(self, xl_file, sheet_name):

        df_sheet = xl_file.parse(sheet_name, skip_rows=1)

        contract_col_mapper = self.create_contract_col_mapper(df_sheet)

        self.create_sub_dataframes(sheet_name, df_sheet, contract_col_mapper, self._volatility, self._volatility_scaling)

    def create_sub_dataframes(self, sheet_name, df_sheet, contract_col_mapper, volatility = 1, volatility_scaling = 10000):
        """

        Create a new dataframe for each contract and store it in self.dictionary_of_contracts
        :param df_sheet:
        :param contract_col_mapper:
        volatility decides whether to add a volatility column or not
        :return:
        """

        # Loop over each contract and and select only the columns of interest
        for contract_name, col_ids in contract_col_mapper.items():

            # this will be true for indices that are in the contract_col_ids relevant to the contract
            col_bools = [True if col_id_ in col_ids else False
                         for col_id_, col in enumerate(df_sheet.columns)]
            # This is creating a sub df of only that contract
            # note that the rows after the last entry are all populated by nans here
            sub_df = df_sheet.loc[:, col_bools]

            # Set the row 1 to be the column (index) of the dataframe
            sub_df.rename(columns=sub_df.iloc[1], inplace=True)

            # remove the first two rows (row 0: the contract name), (row 1: the column names)
            sub_df.drop(sub_df.index[0:2], inplace=True)

            # Remove the rows which are entirely empty (due to the fact that we have sliced this sub_df from a larger df)
            # also removes partially empty rows as these will only cause problems later 
            # and deals with when the day after last trading day is included
            sub_df.dropna(axis=0, how='any', inplace=True)

            # If we dont have any rows left, we just skip
            if sub_df.shape[0] == 0:
                continue


            # Check if the Date column has been explicitly defined, otherwise we need to infer which column it is
            if 'Date' not in sub_df.columns:
                # note that nan != nan
                candidate_columns = [col_ for col_ in sub_df.columns if col_!=col_]
                for candidate_column in candidate_columns:
                    if all(isinstance(cell, datetime.datetime) for cell in sub_df[candidate_column].dropna()):
                        sub_df.rename(columns={candidate_column: 'Date'}, inplace=True)

            # Create a temporary column called datetime_obj into which we will store the datetime objects assosciated
            # with the date. We do this because we 'clean' the date by transforming it into the UK format, but we still
            # want access to the datetime object in order to make use of the datetime library utilities
            sub_df['datetime_obj'] = sub_df['Date']

            # Clean the date column
            try:
                sub_df['Date'] = sub_df['Date'].apply(lambda x: x.strftime('%d/%m/%Y'))

            # If we cant parse the date, skip this row
            except:
                continue
            # Add the volatility column, applying a scaling factor: volatility_scaling
            # choosing axis = 1 in .apply goes row by row so that we can index the columns in each row
            sub_df["Volatility"] = \
                    sub_df.apply(func = lambda row: \
                                      (((np.log(row["px_high"]) - np.log(row["px_low"])) ** 2 \
                                       / (4 * np.log(2))) * volatility_scaling) \
                                       , axis = 1)
            # Adding spread between futures and spot last prices as a percentage of the spot last price
            if "px_spot_last" in sub_df.columns:
                sub_df["Spread"] = \
                        sub_df.apply(func = lambda row: \
                                          ((row["px_last"] - row["px_spot_last"])/row["px_spot_last"]) \
                                              , axis = 1)
            else:
                continue
            ### adding columns of explanatory variables squared. Note that the m ones are done
            # later in filter last months as that is where the maturity column is added
            sub_df["volume2"] = \
                    sub_df.apply(func = lambda row: \
                                      (row["volume"] ** 2),axis = 1)
            
                        
            sub_df["Open Int2"] = \
                    sub_df.apply(func = lambda row: \
                                      (row["Open Int"] ** 2),axis = 1)
            
            

            # Set the date column as the index of the df, this is so we can later concatenate different dfs on this index
            sub_df.set_index('Date', inplace=True)
            # Update dictionary_of_contracts
            self.dictionary_of_contracts[sheet_name][contract_name] = sub_df


    def save_dataframes(self, sub_folder_name=None):
        """

        A utility to save the dataframes to disk
        :param sub_folder_name: (str) we may specifiy a sub folder within self._folder_path to save the dataframes in
        :return:
        """

        # Loop over the contracts, but need to access each sheet first
        for sheet_name in self.dictionary_of_contracts.keys():
            
            for contract_name, contract_df in self.dictionary_of_contracts[sheet_name].items():

                # Create a copy since we do not want to risk changing the dataframe
                contract_df = contract_df.copy()

                # Determine where we want to save the dataframe
                folder_path = self._folder_path if sub_folder_name is  None else os.path.join(self._folder_path, sub_folder_name)

                # Create this directory if it does not exist
                os.makedirs(folder_path, exist_ok=True)

                # Save the dataframe
                file_path = os.path.join(folder_path, '{}.csv'.format(contract_name))
                contract_df.to_csv(file_path, index=True)

    def filter_last_months(self, num_months_dict=None, maturity = 1):
        """

        Filter the data to keep only the relevant months

        Excluding the last day if there is missing data

        If the date ends in March, June or September, we keep the last three months
        If the date ends in December, we keep the last four months
        :param num_months_dict:
            e.g.
                {
                'December': 4,
                'March': 3
                }
        maturity decides whether to add a maturity column or not (measured in days)
        :return:
        """

        # We use a defaultdict since most contracts require a backlog of 2 months
        months_to_keep = defaultdict(lambda : 2)

        # As per the requirements, we update March, June, September and December
        default_values = {
            12: 4,
            3: 3,
            6: 3,
            9: 3
        }

        months_to_keep.update(default_values)

        # Also update this with any user overriden values
        if num_months_dict is not None:
            months_to_keep.update(num_months_dict)


        temp_dictionary_of_contracts = defaultdict(dict)

        # Loop through the contracts and keep only the relevant entries
        for sheet_name in self.dictionary_of_contracts.keys():
            
            for contract_name, contract_df in self.dictionary_of_contracts[sheet_name].items():

                # Isolate the last row and check what month it was in
                last_entry = contract_df.iloc[-1]
                month = last_entry['datetime_obj'].month

                # Calculate the earliest allowed date
                earliest_allowed_date = contract_df.iloc[-1]['datetime_obj'] - pd.DateOffset(months=months_to_keep[month])
                earliest_allowed_date = earliest_allowed_date.to_pydatetime()

                # Remove any entries whose dates precede this
                filtered_df = contract_df[contract_df['datetime_obj'] > earliest_allowed_date]
            
                
            
                # add in maturity column
                # filtered_df['datetime_obj'].apply(maturity_calc, ...) passes in the datetime_obj
                # column as the first argument in the func
                if maturity == 1:
                    filtered_df["Maturity"] = \
                        filtered_df.apply(func = lambda row: (last_entry['datetime_obj'] - row['datetime_obj']).days\
                                       , axis = 1)
                else:
                    break
                
                filtered_df["Maturity2"] = \
                    filtered_df.apply(func = lambda row: \
                                      (row["Maturity"] ** 2),axis = 1)
                
                filtered_df["m_oi"] = \
                    filtered_df.apply(func = lambda row: \
                                      (row["Maturity"] * row["Open Int"]),axis = 1)
                filtered_df["m_v"] = \
                    filtered_df.apply(func = lambda row: \
                                      (row["volume"] * row["Maturity"]),axis = 1)
                

                # We no longer need the datetime_obj (we still have the date in the index)
                # since we are done with any date related calculations
                # filtered_df.drop(columns=['datetime_obj'], inplace=True)
                temp_dictionary_of_contracts[sheet_name][contract_name] = filtered_df

        # Update self.dictionary_of_contracts
        self.dictionary_of_contracts = temp_dictionary_of_contracts

    def create_normalised_dataframe(self, indep_vars, dep_var, stats =  0, oi_graphs = 0, start_date='01/01/1900', end_date='31/12/2100'):
        """

        Combine information from all dataframes into a single dataframe, where the date is consistent
        :return:
        """

        # There will be ambiguity if we were just to plainly combine the dataframes, since each dataframe shares
        # column names, we need to keep track of where each column came from

        temp_dictionary_of_contracts = defaultdict(dict)

        # Loop over the contracts
        for sheet_name in self.dictionary_of_contracts.keys():
            for contract_name, df in self.dictionary_of_contracts[sheet_name].items():

                # Provide a 'super header' for the columns associated with this particular contract
                df = pd.concat({contract_name: df}, axis=1, names=["Contract Name", ""])
                temp_dictionary_of_contracts[sheet_name][contract_name] = df

        self.dictionary_of_contracts = temp_dictionary_of_contracts

        # Concatenate all of the dataframes
        normalised_df = pd.concat(
            [
                df
                for sheet_name in self.dictionary_of_contracts.keys()
                for contract_name, df in self.dictionary_of_contracts[sheet_name].items()
            ], 
            axis=1
        )
        
        # only include contracts within the start and end dates
        normalised_df[('metadata', 'datetime-object')] = normalised_df.index.map(
            lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))

        normalised_df = normalised_df[
            (normalised_df[('metadata', 'datetime-object')] > datetime.datetime.strptime(start_date, '%d/%m/%Y'))
            &
            (normalised_df[('metadata', 'datetime-object')] < datetime.datetime.strptime(end_date, '%d/%m/%Y'))
        ]
        
        # Save this combined dataframe
        normalised_df.to_csv(os.path.join(self._folder_path, 'normalised_df.csv'))
        
        # access the normalised df and drop the columns that only contain nan (due to the date filtering)
        self._normalised_df = normalised_df.dropna(1, "all")
        
        ### do the linear regression
        if stats == 1:
            for sheet_name in self.dictionary_of_contracts.keys():
                list_rows = []
                for contract_name, df in self.dictionary_of_contracts[sheet_name].items():
                    # need to pass a tuple to index in the df as it's a multiindex dataframe 
                    tuple_indep_vars = [(contract_name, i_var) for i_var in indep_vars] # note contract name won't change in df
                    X = df[tuple_indep_vars] ## indep_vars is already a list, which is was the arg needs
                    X = X.astype(float) # converting all lines to float to not crash
                    y = df[(contract_name,dep_var)] # also need to index with a tuple
                    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

                    # Note the difference in argument order
                    model = sm.OLS(y.values, X.values).fit() # need .values to go df --> array
                    predictions = model.predict(X)
                
                    # populating dict with data for csv
                    row = dict()
                    row["Contract"] = contract_name
                    for i_var, i_coeff in zip(indep_vars, model.params[1:]): # assuming returning things in the same order as passed in
                        row["{}_coeff".format(i_var)] = i_coeff
                    for i_var, i_pvalue in zip(indep_vars, model.pvalues[1:]): # assuming returning things in the same order as passed in
                        row["{}_pvalue".format(i_var)] = i_pvalue
                    row["Adjusted_Rsquared"] = model.rsquared_adj
                
                    list_rows.append(row)
                    
                    # making filenames
                    path_to_folder = os.path.join(self._folder_path, "Stats_Data")
                    os.makedirs(path_to_folder, exist_ok=True)
                    file_name = "{}_{}_{}.csv".format(sheet_name,dep_var,"-".join(indep_vars))
                    stats_df = pd.DataFrame(list_rows)
                    
                    # Making significance columns
                    for i_var in indep_vars: # assuming returning things in the same order as passed in
                        stats_df["{}_sig".format(i_var)] = stats_df["{}_pvalue".format(i_var)].apply(
                            lambda p: 1 if p < 0.05 else 0)
                    # Making summary stats of the significance and avg adj R^2
                    row_percentage = dict()
                    row_count = dict()
                    rsquared_adj_avg = dict()
                    rsquared_adj_avg["Adjusted_Rsquared"] = stats_df["Adjusted_Rsquared"].mean()
                            
                    for i_var in indep_vars:
                        row_percentage["{}_sig".format(i_var)] = \
                        sum(stats_df["{}_sig".format(i_var)])/len(stats_df["{}_sig".format(i_var)]) # can use sum as all sig ones are 1
                    for i_var in indep_vars:
                        row_count["{}_sig".format(i_var)] = sum(stats_df["{}_sig".format(i_var)])
                    summary_rows = [rsquared_adj_avg, row_percentage, row_count]
                    summary_df = pd.DataFrame(summary_rows)
                    combined_df = pd.concat([stats_df, summary_df])
                    
                        
                    combined_df.to_csv(os.path.join(path_to_folder,file_name), index = False)
                    
        # plot oi graphs
        if oi_graphs == 1:

            # numpy arrawy of the dates (which is the index) column
            dates = np.array([self._normalised_df.index])

            def get_highest_oi(row):
                oi_columns = {k: v for (k, v) in row.items() if k[1] == 'Open Int' and not np.isnan(v)}
                max_oi = max(oi_columns.values())
                best_contract_list = [k for (k, v) in row.items() if v == max_oi]
                best_contract = best_contract_list[0][0]
                return best_contract

            def sum_oi(row):
                oi_columns = {k: v for (k, v) in row.items() if k[1] == 'Open Int' and not np.isnan(v)}
                sum_oi = sum(oi_columns.values())

                return sum_oi


            self._normalised_df['HighestOI'] = self._normalised_df.apply(get_highest_oi, axis=1)
            self._normalised_df['SumOI'] = self._normalised_df.apply(sum_oi, axis=1)

            ### total open int vs date graph:

            # The columns of normalised_df are tuples ("contract_name", "volume, Open Int etc")
            # ideally, want to create a oi_df with the date as the index (as it is for normalised_df)
            # and all of the "Open Int" 's as columns [maybe need to keep the contract name attached somehow though
            # to be able to list the dates when the contract which has the largest open interest shifts to next contract,\ but idk if the name is needed for that]
            # Need to then make an array where each entry is the sum of the open ints for that day
            # plot this against the array of dates

            fig, ax = plt.subplots()
            fig.set_size_inches(19, 14)
            ax.tick_params(axis='x', labelrotation=45)

            oi_columns = [col for col in self._normalised_df.columns if col[1] == 'Open Int']
            for column in oi_columns:

                contract_name, metric = column
                

                ax.plot(self._normalised_df[column],  label=contract_name.replace("Curncy",""))

            ax.plot(self._normalised_df['SumOI'], label='Sum')
            every_nth = 30
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)

            fig.subplots_adjust(bottom=0.5)
            ax.legend(loc= "upper left",  ncol=int(len(oi_columns)/5),\
                      fontsize = 9, frameon = 0)
            #ax.legend(loc=8,  ncol=int(len(oi_columns)/6),bbox_to_anchor=(0.5, -0.75), frameon = 0)
            ax.tick_params(bottom=False)
            
            plt.rc('axes', labelsize= "x-large")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel("Open Interest")
            plt.xlabel("Date")
            plt.savefig(os.path.join(self._folder_path, '{}.png'.format(sheet_name)),dpi = 500,bbox_inches= "tight")
            plt.show()

            ### individual contract open int vs date

            # add 4 months of 0s to the start of each contract to backlog (plot first without this to see if ok)
            # plot (on same graph) each column of oi_df (so each contract's oi vs date) and use the column (contract) name as label

            ### list dates where the contract with the largest oi shifts from one contract to the next
            current_highest_oi = None
            changes = dict()
            for date, row in self._normalised_df[['HighestOI']].iterrows():
                value = row['HighestOI'].values[0]
                if value != current_highest_oi:
                    changes[date] = {
                        'From': current_highest_oi,
                        'To': value
                    }
                    current_highest_oi = value

            changes_df = pd.DataFrame.from_dict(changes, orient='index')
            changes_df.to_csv(
                os.path.join(self._folder_path, 'changes_{}.csv'.format(sheet_name))
            )

            # these will be the "valley" intercepts on the plot of the contracts
            
                
            

    def process(self, dep_var, indep_vars, maturity = 1, volatility = 1, volatility_scaling = 10000, stats = 0,
                oi_graphs = 0, start_date='01/01/1900', end_date='31/12/2100'):
        """
        

        Parameters
        ----------
        dep_var : str
            DESCRIPTION.
        indep_vars : list of str
            the regressors in your cbc spec
        maturity : TYPE, optional
            DESCRIPTION. The default is 1.
        volatility : TYPE, optional
            DESCRIPTION. The default is 1.
        volatility_scaling : TYPE, optional
            DESCRIPTION. The default is 10000.
        stats : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        # save params to pass in to methods later
        self._maturity = maturity 
        self._volatility = volatility
        self._volatility_scaling = volatility_scaling
        self._stats = stats
        self._oi_graphs = oi_graphs
        self._start_date = start_date
        self._end_date = end_date

        self.read_excel()

        self.save_dataframes(sub_folder_name='pre_filtering')
        
        self.filter_last_months(num_months_dict=self._num_months_dict, maturity = maturity)

        self.create_normalised_dataframe(indep_vars, dep_var, stats = self._stats, oi_graphs= self._oi_graphs, start_date=start_date, end_date=end_date)
        self.save_dataframes(sub_folder_name='post-filtering')
        
save_folder_path = '/Users/ali/OneDrive - Imperial College London/UROP 2021/OI Graphs'
path_to_excel = '/Users/ali/OneDrive - Imperial College London/UROP 2021/OI Graphs/★211128-1212 Bloomberg_API_v5.5.xlsx'

# running for BTC

num_months_dict = None # use this argument to use default values for num months to keep at the end

# set to keep last 50 months to keep all data for the oi graphs
num_months_dict_oigraphs = {
    1: 50,
    2: 50,
    3: 50,
    4: 50,
    5: 50,
    6: 50, 
    7: 50,
    8: 50,
    9: 50,
    10: 50,
    11: 50,
    12: 50,
}

object_ = ContractsDataHandler(
    path=path_to_excel,
    folder_path=save_folder_path,
    sheets_to_read=["BTC"],
    num_months_dict = num_months_dict_oigraphs
)

object_.process(
    dep_var = "Spread",
    indep_vars = ["Maturity", "Open Int", "volume"],
    stats =0,
    oi_graphs=1,
    start_date='01/01/2020',
    end_date='31/12/2100'
)


