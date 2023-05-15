from tensorflow.keras import backend as K
import os
import numpy
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split


def r2_loss(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    return tf.maximum(K.epsilon(), 1 - r2)


def r2_weighted(y_true, y_pred):
    try:
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        r2 = 1 - ss_res/(ss_tot + K.epsilon())
        return tf.maximum(K.epsilon(), 1 - r2)
        # weight = 1 / K.abs(y_true - 1)  # assign higher weight to values closer to 1
        # return K.mean(r2 * weight, axis=-1)
    except AttributeError:
        ss_res = numpy.sum((y_true - y_pred)**2)
        ss_tot = numpy.sum((y_true - numpy.mean(y_true))**2)
        r2 = 1 - ss_res/(ss_tot + numpy.finfo(numpy.float32).eps)
        return max(numpy.finfo(numpy.float32).eps, 1 - r2)


def main():
    """Entry point."""
    # Load US gross output by industry
    us_gross_output_1997_2022 = './data/US Gross Output by Industry 1997-2022.csv'
    us_gross_output_1960_1997 = './data/US Gross Output by Industry 1960-1997.csv'
    us_gross_output_table = None
    for table_path in [us_gross_output_1997_2022, us_gross_output_1960_1997]:
        local_table = pandas.read_csv(
            table_path, skiprows=[0, 1, 2])
        local_table.dropna(subset=['1997'], inplace=True)
        if us_gross_output_table is None:
            us_gross_output_table = local_table
        else:
            us_gross_output_table = pandas.merge(
                us_gross_output_table, local_table, left_index=True, right_index=True,
                suffixes=('', '_dup'))
    us_gross_output_table = us_gross_output_table.rename(columns={'Unnamed: 1': 'Industry'})
    us_gross_output_table = us_gross_output_table.filter(regex='^(?!.*_dup$)')
    us_gross_output_table = us_gross_output_table.set_index(us_gross_output_table.columns[1])
    us_gross_output_table = us_gross_output_table.drop('Line', axis=1)
    us_gross_output_table = us_gross_output_table.replace('...', 0)

    #print(us_gross_output_table)

    co2_emissions_table = pandas.read_csv(
        './data/annual-co2-emissions-per-country.csv')
    co2_emissions_table['Year'] = co2_emissions_table['Year'].astype(str)
    co2_emissions_table = co2_emissions_table[
        co2_emissions_table['Year'].isin(set(us_gross_output_table.columns))]
    co2_emissions_table.dropna(subset=['Code'], inplace=True)
    co2_emissions_table = co2_emissions_table.pivot(
        index='Entity', columns='Year', values=co2_emissions_table.columns[3])
    co2_emissions_table.dropna(inplace=True)

    country_names = co2_emissions_table.index.values
    industry_names = list(us_gross_output_table.index.values)
    n_industries = len(industry_names)

    year_list = list(sorted(us_gross_output_table.columns))
    n_years = len(year_list)

    # n_continuous_years = 5
    # n_layers = 1
    # lstm_density = 4
    # dense_density = 32

    n_continuous_years = 10
    n_layers = 0
    lstm_density = n_continuous_years
    dense_density = 3
    kernel_regular = 0.8
    learning_rate = 0.0001

    # X is the input, i.e. co2 emissions from countries plus the original gdp
    # Y is the output, i.e. gross output from US industries
    # X -> array of n_countries (n_years timesteps) + n_industries elements
    #X = numpy.empty((n_years-n_continuous_years, n_countries, *n_continuous_years+n_industries, 1))
    Y = [] # numpy.empty((n_years-n_continuous_years, n_industries))
    X = []
    for start_index in range(1, n_years-n_continuous_years):
        year_slice = year_list[start_index:n_continuous_years+start_index]
        print(f'working on {year_slice}')

        co2_slice = co2_emissions_table[year_slice]
        co2_country_arrays = [row.reshape(-1, 1) for row in co2_slice.values]
        #us_gross_array = us_gross_output_table[year_slice[0]].values.astype(float)
        input_row = co2_country_arrays # + [us_gross_array]
        X.append(input_row)
        Y.append(
            us_gross_output_table[year_slice[-1]].values.astype(float)-
            us_gross_output_table[year_slice[-2]].values.astype(float))

    Y = numpy.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    X_train_swizzle = [
        [X_train[j][i] for j in range(len(X_train))]
        for i in range(len(X_train[0]))]

    X_test_swizzle = [
        [X_test[j][i] for j in range(len(X_test))]
        for i in range(len(X_test[0]))]

    print(len(X_train_swizzle[0]))

    country_lstm_layers = []
    input_layers = []
    for country_name in country_names:
        input_layers.append(
            Input(shape=(n_continuous_years, 1)))
        country_lstm_layers.append(tf.keras.layers.Dropout(rate=0.4)(
            BatchNormalization()(LSTM(
                lstm_density, input_shape=(
                    n_continuous_years, 1),
                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                activation='linear',
                kernel_regularizer=tf.keras.regularizers.l2(kernel_regular),
                )(input_layers[-1]))))

    #starting_gdp = Input(shape=(n_industries,))
    #input_layers.append(starting_gdp)
    #merged = concatenate(country_lstm_layers + [starting_gdp])
    merged = concatenate(country_lstm_layers)

    #rest_of_layers = BatchNormalization()(merged)

    # Define dense layers after the LSTM layers
    for i in range(n_layers):
        merged = Dense(
            dense_density, activation='linear',
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regular),)(merged)

    # Define output layer
    output_layer = Dense(
        n_industries, activation='linear',
        kernel_initializer=tf.keras.initializers.glorot_uniform())(merged)

    model = Model(inputs=input_layers, outputs=output_layer)


    model.compile(
        loss='mse',
        #loss=r2_weighted,
        #optimizer=tf.keras.optimizers.Adam(clipnorm=0.5))
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),)
        #optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='model_checkpoint_{epoch:05d}',
    #     save_freq='epoch',
    #     verbose=1,
    # )

    # def r2_analysis(model):
    #     predictions = model.predict(X_test)
    #     r2 = r2_score(y_test, predictions)
    #     print(f'R2 so far: {r2}')

    # r2_callback = tf.keras.callbacks.LambdaCallback(
    #     on_epoch_end=lambda epoch, logs: r2_analysis(model))

    csv_logger = CustomCSVLogger('training_log.csv', X_test_swizzle, y_test)

    print(y_train[0])

    os.makedirs('value_tables', exist_ok=True)
    for stage in range(20):
        model.fit(
            x=X_train_swizzle, y=y_train, epochs=200, batch_size=1000000,
            verbose=1, callbacks=[csv_logger],
            use_multiprocessing=True, workers=os.cpu_count())

        for year_to_analyze in range(n_continuous_years+1960, 2020):
            try:
                year_index = year_to_analyze-1960
                print(year_index)
                base_year_x = [[X[year_index][i]] for i in range(len(X[0]))]
                base_predictions = model.predict(base_year_x)
                with open(f'value_tables/value_table_model_{stage}_{year_to_analyze}.csv', 'w') as value_table:
                    value_table.write('Incremental value of in CO2 reduction')
                    value_table.write(f'proportion of original CO2,' + ','.join([f'"{name}"' for name in industry_names]) + '\n')
                    for percent_reduction in numpy.arange(10, 201, 10) / 100.0:
                        reduced_year_x = [
                            [input_array[0]*percent_reduction] for input_array in base_year_x[:-1]] + [base_year_x[-1]]
                        reduced_predictions = model.predict(reduced_year_x)

                        value = [
                            reduced_predictions[0][i]
                            #base_predictions[0][i] - reduced_predictions[0][i]
                            for i in range(len(base_predictions[0]))]
                        value_table.write(f'{percent_reduction-1},' + ','.join([str(x) for x in value]) + '\n')
            except IndexError:
                continue


class CustomCSVLogger(CSVLogger):
    def __init__(self, filename, X_test, y_test, separator=',', append=False):
        super().__init__(filename, separator=separator, append=append, )
        # Define your custom headers here
        self.X_test = X_test
        self.y_test = y_test
        self.loss_list = []
        self.r2_list = []
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Call the parent method to log the default metrics
        super().on_epoch_end(epoch, logs)

        logs = logs or {}
        if self.keys is None:
            self.keys = sorted(logs.keys())

        # Open the CSV file and append the custom metrics
        with open(self.filename, 'a', newline='') as f:
            predictions = self.model.predict(x=self.X_test)
            try:
                #r2 = r2_score(self.y_test, predictions)
                r2 = 1-r2_weighted(self.y_test, predictions)
            except ValueError:
                r2 = -99
            loss = logs['loss']
            line = f'{loss},{r2}\n'
            print(line)
            f.write(line)
            self.loss_list.append(loss)
            self.r2_list.append(r2)
            self.epoch += 1

        # Create data for two lines
        x = list(range(self.epoch))

        # Create figure and axes
        fig, ax1 = plt.subplots()

        # Plot first line on first axis
        ax1.plot(x, self.loss_list, color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create second axis and plot second line on it
        ax2 = ax1.twinx()
        ax2.plot(x, self.r2_list, color='red')
        ax2.set_ylabel('R2', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        os.makedirs('graphs', exist_ok=True)
        plt.savefig(f'graphs/{self.epoch:06d}_{r2:.4f}.png')
        plt.close()


if __name__ == '__main__':
    main()
