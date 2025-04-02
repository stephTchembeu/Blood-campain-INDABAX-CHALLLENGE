from libs import *


## Comon functions 
# Function to update active tab
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name

# show the title of the tab section
def show_title(title,size):
    st.markdown(f"<h1 style='font-size: {size}px;color:rgb(128,4,0)'>{title}</h1>", unsafe_allow_html=True)

# show title for a section i the sideBar
def set_sidebar_title(title):
    st.sidebar.markdown(f"<h2 style='color: rgb(128,4,0); font-size: 20px;'>{title}</h2>", unsafe_allow_html=True)

# show a message to handle error
def indication_message(message):
    st.markdown("""
        <style>
            .custom-container {
                background-color: rgba(255,178,102,0.3); /* Light grey background */
                padding: 40px;
                border-radius: 15px; /* Rounded corners */
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2); /* Shadow effect */
                text-align: center; /* Centered text */
                width: 80%; /* Width of the container */
                margin: auto; /* Center it horizontally */
                font-size: 20px;
                font-weight:bold;
                color:rgb(255,102,102);
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="custom-container">{message}</div>', unsafe_allow_html=True)

# donor distribution functions
# here we generated color for the map
def get_gradient_color(value, min_value, max_value):
    """Returns an RGB color between white (255,255,255) and burgundy (128,4,0)"""
    ratio = (value - min_value) / (max_value - min_value) if max_value > min_value else 0
    r = int(255 - (127 * ratio))  # 255 → 128
    g = int(255 - (251 * ratio))  # 255 → 4
    b = int(255 - (255 * ratio))  # 255 → 0
    return f"rgb({r},{g},{b})"


# eligibility functions     
# get the parameter for the prediction
def get_eligibility_observations_param(data,observered_column):
    # get the eligibility column
    el_col = st.sidebar.multiselect("Choose the health condition:", data.columns[data.columns == "ELIGIBILITE AU DON"])
    try:
        if el_col:  # Ensure a column is selected
            eligibility_types = data[el_col[0]].unique()
            selected_el = st.multiselect("Filter eligibility status:", eligibility_types, default=eligibility_types[0])
        else:
            selected_el = []
    except:
        with tabs[1]:
            st.warning("fill the eligibility form section on the sidebar to continue")
    # Select Health Condition Columns to see their impact on eligibility
    selected_conds = st.multiselect("🩺 Select health conditions:", observered_column)
    return el_col,selected_el,selected_conds
# Display Key Statistics
def printer(list_):
    if len(list_) == 1:
        return f"{list_[0]}"
    elif len(list_) == 2:
        return f"{list_[0]} and {list_[1]}"
    else:
        return f"{list_[0]}, {list_[1]} and {list_[2]}"
# show the plots for the eligibility
def show_el_plots(conds,data,el_col):
    # Arrange plots in rows with max 3 per row
            num_conditions = len(conds)
            num_cols = min(num_conditions, 3)
            
            rows = [conds[i:i+num_cols] for i in range(0, num_conditions, num_cols)]
            
            for row in rows:
                cols = st.columns(len(row))
                
                for i, condition in enumerate(row):
                    with cols[i]:
                        st.markdown(f"<p style='font-size: 12px; font-weight: bold;'>{condition} impact on eligibility</p>", unsafe_allow_html=True)
                        
                        # Grouping data
                        condition_eligibility_counts = data.groupby([condition, el_col[0]]).size().reset_index(name="count")
                        
                        # Ensure there is data to plot
                        if condition_eligibility_counts.empty:
                            st.warning(f"No data available for {condition}.")
                            continue
                        
                        # Display Chart with fixed bar color and height
                        fig = px.bar(
                            condition_eligibility_counts, 
                            x=condition,  # X-axis: health condition values
                            y="count",  # Y-axis: count of donors
                            color=el_col[0],  # Color: Eligibility Status
                            barmode="stack"
                        )

                        # Manually set the bar color and figure height
                        fig.update_traces(marker=dict(color="rgb(128,4,0)"))
                        fig.update_layout(height=300)  # Set height to 300px
                        # show the plot
                        st.plotly_chart(fig, use_container_width=True)



## Donors retention functions
# plot the pie and trend of the first time donor and none first time donors
def show_recurrent_pie_trend(data,filtered):
    if st.session_state.has_donated_before_column in data.columns:
                # Clean up the response data (assuming "oui" means "yes" and anything else means "no")
                filtered[st.session_state.has_donated_before_column] = filtered[st.session_state.has_donated_before_column].fillna("non")
                filtered['Has_Donated_Before'] = filtered[st.session_state.has_donated_before_column].str.lower().apply(
                    lambda x: "Has donated before" if x == "oui" else "First-time donor"
                )
                
                # Plot as a donut chart
                donation_history = filtered['Has_Donated_Before'].value_counts().reset_index()
                donation_history.columns = ['status', 'count']
                
                # plot the pie of the proportion of first time donor and none first time donors
                fig_donut = px.pie(
                    donation_history,
                    values='count',
                    names='status',
                    title="Previous Donation History",
                    color='status',
                    hole=0.4,  # Donut chart
                    color_discrete_map={
                        "First-time donor": "rgb(128,4,0)",
                        "Has donated before": "rgb(23,158,14)"
                    }
                )
                fig_donut.update_layout(legend_orientation="h", legend_y=-0.2)
                
                # Previous donation history trends over time
                monthly_history = filtered.groupby([filtered[st.session_state.date_column].dt.to_period("M"), "Has_Donated_Before"]).size().reset_index()
                monthly_history.columns = ['month', 'status', 'count']
                monthly_history['month'] = monthly_history['month'].astype(str)
                
                # plot the trend of frequence of donation amoung first time donors and none first time donors
                fig_trend = px.line(
                    monthly_history,
                    x='month',
                    y='count',
                    color='status',
                    title="First-time vs Returning Donors Over Time",
                    labels={'month': 'Month', 'count': 'Number of Donors', 'status': 'Donor Status'},
                    color_discrete_map={
                        "First-time donor": "rgb(128,4,0)",
                        "Has donated before": "rgb(23,158,14)"
                    }
                )
                fig_trend.update_layout(xaxis_tickangle=-45)
                
                # Display side by side with ratio 0.4 for donut and 0.6 for trend
                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    st.plotly_chart(fig_donut, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_trend, use_container_width=True)
# donation intervals for repeat donors
def donation_intervale(data,filtered):
    if st.session_state.last_donation_column in data.columns:
                filtered["Donation_Interval"] = (filtered[st.session_state.date_column] - filtered[st.session_state.last_donation_column]).dt.days
                repeat_donors = filtered[filtered["Donation_Interval"].notna()]
                
                # Filter out unreasonable intervals (negative or extremely large)
                reasonable_intervals = repeat_donors[
                    (repeat_donors["Donation_Interval"] > 0) & 
                    (repeat_donors["Donation_Interval"] < 1000)  # Assume donations over ~3 years are data errors
                ]
                
                # 
                if not reasonable_intervals.empty:
                    show_title("Time Between Donations",20)
                    # Create histogram of donation intervals
                    fig_hist = px.histogram(
                        reasonable_intervals,
                        x="Donation_Interval",
                        nbins=30,
                        title="Distribution of Time Between Donations",
                        labels={"Donation_Interval": "Days Between Donations", "count": "Number of Donors"},
                        color_discrete_sequence=["#C62828"]
                    )
                    
                    # Add median and mean lines
                    median_interval = reasonable_intervals["Donation_Interval"].median()
                    mean_interval = reasonable_intervals["Donation_Interval"].mean()
                    
                    fig_hist.add_vline(x=median_interval, line_dash="dash", line_color="black", annotation_text=f"Median: {median_interval:.0f} days")
                    fig_hist.add_vline(x=mean_interval, line_dash="dot", line_color="blue", annotation_text=f"Mean: {mean_interval:.0f} days")
                    fig_hist.add_vline(x=56, line_dash="dash", line_color="green", annotation_text="Min. Safe: 56 days")
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Display median/mean intervals
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Median Time Between Donations", f"{median_interval:.0f} days")
                    with col2:
                        st.metric("Average Time Between Donations", f"{mean_interval:.0f} days")

                    # Analyze donation intervals by demographic factors
                    donation_by_demographie(reasonable_intervals)                    
# demographie of donors
def donation_by_demographie(intervals):
    st.subheader("Donation Intervals by Demographic")
    # Select demographic to analyze
    demo_options = [col for col in st.session_state.demographic_options if col in intervals.columns]
    if demo_options:
        selected_demo = st.selectbox("Select demographic to analyze donation intervals:", demo_options)
                        
        # Box plot of intervals by demographic
        fig_box = px.box(
            intervals,
            x=selected_demo,
            y="Donation_Interval",
            title=f"Distribution of Donation Intervals by {selected_demo}",
            labels={selected_demo: selected_demo, "Donation_Interval": "Days Between Donations"},
            color=selected_demo
        )
        st.plotly_chart(fig_box, use_container_width=True)



# prediction function
# show the prediction form to get data for prediction     
def show_pred_form():
    with st.form(key='input_form'):
            st.input_data = st.text_area("Input Data (in JSON format)", "")
            st.pred_submit_button = st.form_submit_button(label='Submit')

# show prediction result
def try_pred():
        if st.pred_submit_button:
            try:
                # We convert the input data to a dictionary
                st.input_data_dict = json.loads(st.input_data)
                # We send a POST request to our Flask API lauched previously
                st.response = requests.post("http://127.0.0.1:5001/predict", json=st.input_data_dict)
                # Display the response
                if st.response.status_code == 200:
                    st.success("Request successful!")
                    st.json(st.response.json())
                else:
                    st.error(f"Request failed with status code {st.response.status_code}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")


