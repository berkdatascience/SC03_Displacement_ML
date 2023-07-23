import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models and scaler
scaler = joblib.load('scaler.joblib')
models = {}
output_cols = ['axial_disp', 'radial_disp']
for output_col in output_cols:
    models[output_col] = joblib.load(f'{output_col}_model.joblib')

def main():
    st.title('Displacement Prediction')
    st.write('Enter the input values to predict axial_disp and radial_disp.')

    # Create input fields for each feature
    input_data = {}
    for col in ['htc', 'flow_temperature', 'x_coordinate', 'y_coordinate',
                'y_coordinate_square', 'elastic_modulus', 'poisson_ratio',
                'thermal_expansion', 'thermal_conductivity', 'specific_heat',
                'fluid_pressure', 'fluid_heat_flux', 'idle', 'mto']:
        input_data[col] = st.number_input(col, value=0.0)

    # Predict when the user clicks the "Predict" button
    if st.button('Predict'):
        input_data_df = pd.DataFrame([input_data])
        input_data_scaled = scaler.transform(input_data_df)

        predictions = {}
        for output_col in output_cols:
            predictions[output_col] = models[output_col].predict(input_data_scaled)

        st.subheader('Predicted Displacements:')
        st.write(predictions)

if __name__ == '__main__':
    main()
