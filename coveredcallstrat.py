import streamlit as st
import numpy as np
import plotly.graph_objects as go
from covered_call_strategy import CoveredCallStrategy
import pandas as pd

# Configure page layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def simulate_gbm(S0, drift, volatility, T, N, num_sims):
    """
    Simulate Geometric Brownian Motion for stock prices
    """
    dt = T/N
    weekly_drift = drift * dt
    weekly_vol = volatility * np.sqrt(dt)
    
    Z = np.random.normal(0, 1, size=(num_sims, N))
    
    prices = np.zeros((num_sims, N+1))
    prices[:, 0] = S0
    
    for t in range(1, N+1):
        prices[:, t] = prices[:, t-1] * np.exp(
            (drift - 0.5 * volatility**2) * dt + 
            volatility * np.sqrt(dt) * Z[:, t-1]
        )
    
    return prices

def main():
    st.title("Stock Price Path Simulation with Covered Call Strategy")
    
    # Add detailed explanation
    st.subheader("Mathematical Framework")
    st.markdown("""
    This simulation uses Geometric Brownian Motion (GBM) to model stock price movements. The core of this model lies in its expected return calculation, given by the formula e^(μT) - 1 = e^((drift - 0.5σ²)T) - 1, where drift is the user input annual return (such as 7%) and σ (sigma) represents the volatility input (such as 15%). The -0.5σ² term in this equation exists to adjust for the geometric nature of returns. The actual stock price evolution in each step follows the formula S(t+1) = S(t) * e^((drift - 0.5σ²)dt + σ√dt * Z), where S(t) is the stock price at time t, dt represents the time step (1/52 for weekly intervals), and Z is a random number drawn from a standard normal distribution. In the resulting simulation, the histogram displays the distribution of final returns, which theoretically centers around the expected return and demonstrates a standard deviation approximately equal to the input volatility over the one-year period.
    """)
    
    
    # Sidebar inputs for stock simulation
    st.sidebar.header("Stock Simulation Parameters")
    S0 = st.sidebar.number_input(
        "Initial Stock Price", 
        value=100.0,
        help="The starting price of the stock in dollars"
    )
    drift = st.sidebar.number_input(
        "Annual Drift (%)", 
        value=7.0,
        help="Expected annual return of the stock. For example, 7.0 means a 7% expected annual return"
    ) / 100
    sim_volatility = st.sidebar.number_input(
        "Simulation Volatility (%)", 
        value=15.0,
        help="Your expected actual volatility of the stock over the coming year. This represents how much you think the stock will actually move"
    ) / 100
    weeks = st.sidebar.number_input(
        "Time (weeks)", 
        value=52,
        help="Duration of the simulation in weeks. One year = 52 weeks"
    )
    num_sims = st.sidebar.number_input(
        "Number of Simulations", 
        value=30,
        help="Number of different price paths to simulate. More simulations give more stable results but run slower"
    )

    # Covered Call Parameters with separate implied vol
    st.sidebar.markdown("---")
    st.sidebar.header("Covered Call Parameters")
    num_positions = st.sidebar.number_input(
        "Number of 100-Share Positions", 
        min_value=1, 
        value=4,
        help="Each position represents 100 shares. For example, 4 positions means you're trading 400 shares total"
    )
    implied_volatility = st.sidebar.number_input(
        "Market Implied Volatility (%)", 
        value=20.0,
        help="The volatility implied by current market option prices. This determines the premium you receive for selling calls. Often higher than actual expected volatility"
    ) / 100
    target_delta = st.sidebar.slider(
        "Target Call Delta to Sell", 
        0.05, 0.70, 0.30, 0.01,
        help="Delta of the calls you want to sell. Higher delta means more premium but higher chance of assignment. 0.30 means 30-delta calls"
    )
    weeks_to_expiry = st.sidebar.number_input(
        "Weeks to Expiration", 
        value=4, 
        min_value=1,
        help="How many weeks until the calls you sell expire. Longer dated options give more premium but lock up your position longer"
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)", 
        value=4.5,
        help="The risk-free interest rate used in option pricing, typically based on Treasury yields"
    ) / 100
    vega_scale = st.sidebar.number_input(
        "Vega Scale", 
        value=0.1402, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.01,
        help="Controls how much implied volatility changes with stock price movement. 0 means constant volatility, higher values mean volatility increases more when stock price drops"
    )





    # Convert weeks to years for calculation
    T = weeks / 52
    N = 52  # Fixed number of time steps (weekly)
    
    # Simulate prices
    prices = simulate_gbm(S0, drift, sim_volatility, T, N, num_sims)
    
    # Initialize covered call strategy
    strategy = CoveredCallStrategy(
        num_positions=num_positions,
        target_delta=target_delta,
        weeks_to_expiry=weeks_to_expiry,
        risk_free_rate=risk_free_rate,
        base_volatility=implied_volatility,  # Using implied vol for option pricing
        vega_scale=vega_scale
    )
    
    # Create time points for x-axis (in weeks)
    time_points = np.linspace(0, weeks, N+1)
    
    # Create container for stock price simulation
    st.header("Stock Price Simulation")
    col1, col2 = st.columns(2)
    
    # Stock price paths plot
    fig_stock = go.Figure()
    for i in range(num_sims):
        fig_stock.add_trace(
            go.Scatter(
                x=time_points,
                y=prices[i, :],
                mode='lines',
                opacity=0.5,
                line=dict(width=1),
                name=f'Path {i+1}'
            )
        )
    
    mean_path = np.mean(prices, axis=0)
    fig_stock.add_trace(
        go.Scatter(
            x=time_points,
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path'
        )
    )
    
    fig_stock.update_layout(
        title='Stock Price Paths',
        xaxis_title='Time (weeks)',
        yaxis_title='Stock Price ($)',
        showlegend=False,
        height=500,
        margin=dict(t=30, b=30)  # Adjust margins for better spacing
    )
    
    # Stock returns histogram
    final_returns = (prices[:, -1] - S0) / S0 * 100
    mean_return = np.mean(final_returns)
    std_return = np.std(final_returns)
    
    fig_hist_stock = go.Figure()
    fig_hist_stock.add_trace(
        go.Histogram(
            x=final_returns,
            nbinsx=30,
            name='Stock Returns'
        )
    )
    
    fig_hist_stock.add_vline(x=mean_return, line_color='red', 
                            annotation_text=f'Mean: {mean_return:.2f}%')
    fig_hist_stock.add_vline(x=mean_return + std_return, line_color='green', 
                            line_dash='dash', annotation_text=f'+1σ: {(mean_return + std_return):.2f}%')
    fig_hist_stock.add_vline(x=mean_return - std_return, line_color='green', 
                            line_dash='dash', annotation_text=f'-1σ: {(mean_return - std_return):.2f}%')
    
    fig_hist_stock.update_layout(
        title='Distribution of Stock Returns',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        showlegend=False,
        height=500,
        margin=dict(t=30, b=30)  # Matching margins with stock price plot
    )
    
    # Display stock plots
    with col1:
        st.plotly_chart(fig_stock, use_container_width=True)
    with col2:
        st.plotly_chart(fig_hist_stock, use_container_width=True)
    
####################################################################


    # Calculate metrics from simulation results
    total_trades = 0
    total_assignments = 0
    total_premiums = 0
    total_stock_values = 0


    trade_data = []

# Create covered call strategy plots
    fig_cc = go.Figure()
    portfolio_values = np.zeros((num_sims, len(time_points)))
    portfolio_values_stock = np.zeros((num_sims, len(time_points)))
    portfolio_values_cash = np.zeros((num_sims, len(time_points)))
    
    # Process each simulation path through the strategy
    for sim_idx in range(num_sims):
        strategy = CoveredCallStrategy(
            num_positions=num_positions,
            target_delta=target_delta,
            weeks_to_expiry=weeks_to_expiry,
            risk_free_rate=risk_free_rate,
            base_volatility=implied_volatility,  # Using implied vol for option pricing
            vega_scale=vega_scale
        )
        
        sim_path = prices[sim_idx, :]
        strategy.initialize_positions(sim_path[0])
        
        # Initialize starting portfolio value
        portfolio_values[sim_idx, 0] = num_positions * 100 * sim_path[0]  # Initial stock value, no cash yet
         

        print("NEW SIMULATION --------------------------------------")
        # Track chronologically

        
        current_cash = 0
        for pos_idx, position in enumerate(strategy.positions):
            trade_data.append({
                'Week': 0,
                'Simulation': sim_idx + 1,
                'Position': pos_idx + 1,
                'Stock_Price': sim_path[0],
                'Has_Contract': False,
                'Strike_Price': None,
                'Premium': None,
                'Days_To_Expiry': None,
                'Position_Cash': position.cash_balance,
                'Total_Cash': current_cash,
                'Total_Portfolio_Value': num_positions * 100 * sim_path[0],
                'Was_Assigned': False
            })


        for t in range(1, len(time_points)):
            print("time step is ")
            print(t)
            call_written = False

            # Reset assignment tracking for this timestep

            position_assignments = {pos.position_id: False for pos in strategy.positions}
            # Check for expiring calls and write new ones. check all positions for 
            # seeing if they have expired
            for position in strategy.positions:
                # Check if we have an expiring call
                if position.current_call and position.current_call.expiry_week == t:
                    stock_price = sim_path[t]
                    expiry_result = strategy.check_expiration(position, t, stock_price)
                    if expiry_result and expiry_result.assigned:

                        print("position assigned!")
                        position_assignments[position.position_id] = True
                        # Update cash from assignment that just happened
                        current_cash = sum(p.cash_balance for p in strategy.positions)
                
                # Write new call if needed. only writing one call a week max
                # that way they are staggered to start with
                if not call_written:
                    if not position.current_call:



                        call_written = True
                        new_call = strategy.write_call(position, t, sim_path[t])
                        current_cash = sum(p.cash_balance for p in strategy.positions)

                        print ("call written!")
                        print(new_call.stock_price_at_write)
                        print(new_call.strike)
                        print(new_call.premium)
                        print("sim id:")
                        print (sim_idx)
                        print("position_id:")
                        print(position.position_id)
                        print("position cash balance")
                        print(position.cash_balance)
            
            #--------#

            # Calculate portfolio value at this time point
            stock_value = num_positions * 100 * sim_path[t]
            # save the portfolio value at this point 
            portfolio_values[sim_idx, t] = stock_value + current_cash
            portfolio_values_stock[sim_idx, t] = stock_value
            portfolio_values_cash[sim_idx, t] = current_cash

            # double for loop
            for pos_idx, position in enumerate(strategy.positions):
                trade_data.append({
                    'Week': t,
                    'Simulation': sim_idx + 1,
                    'Position': pos_idx + 1,
                    'Stock_Price': sim_path[t],
                    'Has_Contract': position.current_call is not None,
                    'Strike_Price': position.current_call.strike if position.current_call else None,
                    'Premium': position.current_call.premium if position.current_call else None,
                    'Days_To_Expiry': (position.current_call.expiry_week - t) * 7 if position.current_call else None,
                    'Position_Cash': position.cash_balance,
                    'Total_Cash': current_cash,
                    'Total_Portfolio_Value': portfolio_values[sim_idx, t],
                    'Was_Assigned': position_assignments[position.position_id]
                })

            print("stock value and current cash respectively")
            print(stock_value)
            print(current_cash)

        # end the for t in range here ( the simulations for a given line )
        





        # Plot this simulation path
        fig_cc.add_trace(
            go.Scatter(
                x=time_points,
                y=portfolio_values[sim_idx, :],
                mode='lines',
                opacity=0.5,
                line=dict(width=1),
                name=f'Path {sim_idx+1}'
            )
        )

        for position in strategy.positions:
            for trade in position.trades:
                total_trades += 1
                if trade.assigned:
                    total_assignments += 1
                total_premiums += trade.premium
                total_stock_values += trade.stock_price_at_write
    
#######---- end for loop for strategy (need to fix) #############


    mean_path_cc = np.mean(portfolio_values, axis=0)
    fig_cc.add_trace(
        go.Scatter(
            x=time_points,
            y=mean_path_cc,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path'
        )
    )

    fig_cc.update_layout(
        title='Portfolio Value Paths',
        xaxis_title='Time (weeks)',
        yaxis_title='Portfolio Value ($)',
        showlegend=False,
        height=500,
        margin=dict(t=30, b=30)
    )

    ### HISTOGRAM 2 ########
    # Create strategy returns histogram using actual portfolio values
    portfolio_returns = (portfolio_values[:, -1] - (S0 * num_positions * 100) ) / (S0 * num_positions * 100) * 100
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    
    fig_hist_cc = go.Figure()
    fig_hist_cc.add_trace(
        go.Histogram(
            x=portfolio_returns,
            nbinsx=30,
            name='Strategy Returns'
        )
    )

    
    fig_hist_cc.add_vline(x=mean_return, line_color='red',
                         annotation_text=f'Mean: {mean_return:.2f}%')
    fig_hist_cc.add_vline(x=mean_return + std_return, line_color='green',
                         line_dash='dash', annotation_text=f'+1σ: {(mean_return + std_return):.2f}%')
    fig_hist_cc.add_vline(x=mean_return - std_return, line_color='green',
                         line_dash='dash', annotation_text=f'-1σ: {(mean_return - std_return):.2f}%')
    
    fig_hist_cc.update_layout(
        title='Distribution of Strategy Returns',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        showlegend=False,
        height=500,
        margin=dict(t=30, b=30)
    )
    
    #############
    # Add separation and covered call section
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)  # Add extra space
    

    st.subheader("Strategy Framework")

    st.markdown("""
    This covered call strategy simulation models a portfolio of N x 100 share positions, where each position independently writes covered calls with a target delta (δ). The strike price (K) for each call is determined using the Black-Scholes option pricing model, solving for the strike that gives the desired delta: δ = N(d₁), where d₁ = (ln(S/K) + (r + σ²/2)T)/(σ√T). Each position writes a new call when its previous call expires, with expiration set to X weeks in the future. When a call expires in-the-money (S > K), the shares are called away at the strike price K, and new shares are immediately purchased at the current market price S, using accumulated premium and adding cash if needed. The strategy's total return consists of three components: stock price appreciation/depreciation, premium income from writing calls, and the impact of assignment events (K - S when shares are called away). The cash balance for each position tracks premium income and assignment effects, allowing it to go negative if needed for share repurchases. The resulting distribution of strategy returns typically shows lower volatility than pure stock ownership, with reduced upside potential but enhanced income generation through premium collection.
""")
    st.header("Covered Call Strategy Simulation")
    col3, col4 = st.columns(2)
    
    # Display covered call plots
    with col3:
        st.plotly_chart(fig_cc, use_container_width=True)
    with col4:
        st.plotly_chart(fig_hist_cc, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)  # Add extra space



######### INSIGHTS (NEED TO FIX) #################
# Display strategy insights
    st.markdown("---")
    st.header("Strategy Insights")
    
    # Create four columns for metrics
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    

    
    # Collect metrics across all simulation paths
    
    
    
    if total_trades > 0:
        avg_premium_pct = (total_premiums / total_stock_values) * 100
        assignment_rate = (total_assignments / total_trades) * 100
    else:
        avg_premium_pct = 0
        assignment_rate = 0
    
    avg_closing_cash = np.mean(portfolio_values_cash[:, -1])
    strategy_return = np.mean(portfolio_returns)
    with met_col1:
        st.metric("Average Premium per time period %", f"{avg_premium_pct:.2f}%")
    
    with met_col2:
        st.metric("Assignment Rate", f"{assignment_rate:.1f}%")
    
    with met_col3:
        st.metric("Average Final Cash Balance", f"${avg_closing_cash:,.2f}")
    
    with met_col4:
        st.metric("Mean Covered Call Strategy Return", f"{strategy_return:.1f}%")



######################
    st.markdown("---")
    st.header("Strategy Activity Details")
    
    # Convert trade data to DataFrame
    df = pd.DataFrame(trade_data)
    
    # Format numeric columns
    numeric_cols = ['Stock_Price', 'Strike_Price', 'Premium', 'Position_Cash', 'Total_Cash', 'Total_Portfolio_Value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
   # Add filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_filter = st.selectbox('Filter by Simulation', 
                                options=['All'] + sorted(df['Simulation'].unique().tolist()))
    with col2:
        pos_filter = st.selectbox('Filter by Position', 
                                options=['All'] + sorted(df['Position'].unique().tolist()))
    with col3:
        contract_filter = st.selectbox('Show Contracts', 
                                     options=['All', 'Active Only', 'No Contract Only'])
    with col4:
        assignment_filter = st.selectbox('Show Assignments',
                                       options=['All', 'Assigned Only', 'Not Assigned Only'])
    
    # Apply filters
    filtered_df = df.copy()
    if sim_filter != 'All':
        filtered_df = filtered_df[filtered_df['Simulation'] == sim_filter]
    if pos_filter != 'All':
        filtered_df = filtered_df[filtered_df['Position'] == pos_filter]
    if contract_filter == 'Active Only':
        filtered_df = filtered_df[filtered_df['Has_Contract']]
    elif contract_filter == 'No Contract Only':
        filtered_df = filtered_df[~filtered_df['Has_Contract']]
    if assignment_filter == 'Assigned Only':
        filtered_df = filtered_df[filtered_df['Was_Assigned']]
    elif assignment_filter == 'Not Assigned Only':
        filtered_df = filtered_df[~filtered_df['Was_Assigned']]
    
    # Display filtered DataFrame
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )


    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)  # Add extra space
    

    st.subheader("General takeaways from Christopher")

    st.markdown("""
    The general takeaway here is that selling covered calls is a good idea when current IV is high, causing contracts to be more expensive, but the actual volitility of the stock in the next year is less than that. Selling higher deltas is better when the stock is going down (and that can be seen by changing the drift). Selling the 30 deltas though seems to be a sweet spot that allows you to get some upside with the stock while still collecting good premium.
""")
if __name__ == "__main__":
    main()