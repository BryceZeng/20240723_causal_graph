import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy import CausalModel

# Create node lookup for channels
node_lookup = {
    0: "Demand",
    1: "Call waiting time",
    2: "Call abandoned",
    3: "Reported problems",
    4: "Discount sent",
    5: "Churn",
}

total_nodes = len(node_lookup)

# Create adjacency matrix - this is the base for our graph
graph_actual = np.zeros((total_nodes, total_nodes))

# Create graph using expert domain knowledge
graph_actual[0, 1] = 1.0  # Demand -> Call waiting time
graph_actual[0, 2] = 1.0  # Demand -> Call abandoned
graph_actual[0, 3] = 1.0  # Demand -> Reported problems
graph_actual[1, 2] = 1.0  # Call waiting time -> Call abandoned
graph_actual[1, 5] = 1.0  # Call waiting time -> Churn
graph_actual[2, 3] = 1.0  # Call abandoned -> Reported problems
graph_actual[2, 5] = 1.0  # Call abandoned -> Churn
graph_actual[3, 4] = 1.0  # Reported problems -> Discount sent
graph_actual[3, 5] = 1.0  # Reported problems -> Churn
graph_actual[4, 5] = 1.0  # Discount sent -> Churn


def data_generator(max_call_waiting, inbound_calls, call_reduction):
    """
    A data generating function that has the flexibility to reduce the value of node 0 (Call waiting time) - this enables us to calculate ground truth counterfactuals

    Args:
        max_call_waiting (int): Maximum call waiting time in seconds
        inbound_calls (int): Total number of inbound calls (observations in data)
        call_reduction (float): Reduction to apply to call waiting time

    Returns:
        DataFrame: Generated data
    """

    df = pd.DataFrame(columns=node_lookup.values())

    # Generate a time variable
    df["Time"] = pd.date_range(start="2023-01-01", periods=inbound_calls, freq="H")

    df[node_lookup[0]] = np.random.randint(
        low=10, high=max_call_waiting, size=(inbound_calls)
    )  # Demand
    df[node_lookup[1]] = (df[node_lookup[0]] * 0.5) * (
        call_reduction
    ) + np.random.normal(
        loc=0, scale=40, size=inbound_calls
    )  # Call waiting time
    df[node_lookup[2]] = (df[node_lookup[1]] > 50).astype(int)  # Call abandoned
    df[node_lookup[3]] = (df[node_lookup[2]] * 0.3) + np.random.normal(
        loc=0, scale=10, size=inbound_calls
    )  # Reported problems
    df[node_lookup[4]] = (df[node_lookup[3]] > 5).astype(int)  # Discount sent
    df[node_lookup[5]] = (
        df[node_lookup[1]]
        + df[node_lookup[2]]
        + df[node_lookup[3]]
        + df[node_lookup[4]]
        > 100
    ).astype(
        int
    )  # Churn

    return df


# Generate data
np.random.seed(123)
df = data_generator(max_call_waiting=300, inbound_calls=1200, call_reduction=1.20)


# Setup graph
graph = nx.from_numpy_array(graph_actual, create_using=nx.DiGraph)
graph = nx.relabel_nodes(graph, node_lookup)

# Create SCM
causal_model = gcm.InvertibleStructuralCausalModel(graph)
causal_model.set_causal_mechanism("Demand", gcm.EmpiricalDistribution())  # Root node
causal_model.set_causal_mechanism(
    "Call waiting time", gcm.AdditiveNoiseModel(gcm.ml.create_ridge_regressor())
)  # Non-root node
causal_model.set_causal_mechanism(
    "Call abandoned", gcm.AdditiveNoiseModel(gcm.ml.create_ridge_regressor())
)  # Non-root node
causal_model.set_causal_mechanism(
    "Reported problems", gcm.AdditiveNoiseModel(gcm.ml.create_ridge_regressor())
)  # Non-root node
causal_model.set_causal_mechanism(
    "Discount sent", gcm.AdditiveNoiseModel(gcm.ml.create_ridge_regressor())
)  # Non-root
causal_model.set_causal_mechanism(
    "Churn", gcm.AdditiveNoiseModel(gcm.ml.create_ridge_regressor())
)  # Non-root
gcm.fit(causal_model, df)


####### Inference #######
# Set call reduction to 20%
reduce = 0.20
call_reduction = 1 - reduce
# Causal graph counterfactual
df_counterfactual = gcm.counterfactual_samples(
    causal_model, {"Call waiting time": lambda x: x * call_reduction}, observed_data=df
)
causal_graph = round(
    (df["Churn"].sum() - df_counterfactual["Churn"].sum()) / (df["Churn"].sum()), 3
)

# Define the causal graph
causal_graph = """
digraph {
    Demand -> "Call waiting time";
    Demand -> "Call abandoned";
    Demand -> "Reported problems";
    Demand -> Churn;
    "Call waiting time" -> "Call abandoned";
    "Call waiting time" -> Churn;
    "Call abandoned" -> "Reported problems";
    "Call abandoned" -> Churn;
    "Reported problems" -> "Discount sent";
    "Reported problems" -> Churn;
    "Discount sent" -> Churn;
}
"""

# Create the causal model
model = CausalModel(data=df, treatment="Demand", outcome="Churn", graph=causal_graph)

# Identify the causal effect
identified_estimand = model.identify_effect()
# Estimate the causal effect
estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.linear_regression"
)

# Refute the estimate
refute_results = model.refute_estimate(
    identified_estimand, estimate, method_name="placebo_treatment_refuter"
)
print(estimate)
print(refute_results)
