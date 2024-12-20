from pyvis.network import Network 
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_hex  
import mplcursors
import webbrowser  # Allows opening URLs or local files in the system's default browser
import os  # For handling paths and checking file existence
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')
import tempfile
 


######################################################################################
# Plots the time series of ODE concentrations and abstractions
######################################################################################

def plot_series_ode(time_series, xlabel="Time", ylabel="Concentration", title="Time Series of Concentrations"):
    """
    Plots the time series of ODE concentrations.

    Parameters:
    time_series (pd.DataFrame): Time series with a 'Time' column and species concentrations as columns.
    xlabel (str): Label for the x-axis. Default is "Time".
    ylabel (str): Label for the y-axis. Default is "Concentration".
    title (str): Title of the plot. Default is "Time Series of Concentrations".

    Raises:
    ValueError: If the DataFrame does not contain a 'Time' column.

    Returns:
    None: Displays a line plot for each species in the time series.
    """
    if 'Time' not in time_series.columns:
        # Check if the 'Time' column exists, raise error if not found
        raise ValueError("The DataFrame must include a 'Time' column for time values.")
    
    # Create a new figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Plot size 10x6 inches
    
    # Iterate over the columns to plot each species
    for species in time_series.columns:
        if species != 'Time':  # Skip the 'Time' column
            ax.plot(time_series['Time'], time_series[species], label=species)  # Plot species concentration
    
    # Set axis labels and plot title
    ax.set_xlabel(xlabel)  # Set x-axis label
    ax.set_ylabel(ylabel)  # Set y-axis label
    ax.set_title(title)    # Set the title of the plot
    ax.grid()              # Enable grid on the plot
    ax.legend()            # Add a legend for species
    
    plt.tight_layout()  # Adjust layout to fit all elements
    plt.show()          # Display the plot

######################################################################################## 

def plot_abstraction_size(abstract_time_series, xlabel="Time", ylabel="Number of Species", title="Number of species per abstraction over time", marker='o', label="Abstraction Size"):
    """
    Plots the number of abstractions over the time series.

    Parameters:
    abstract_time_series (pd.DataFrame): Time series with a 'Time' column and a column for species abstractions.
    xlabel (str): Label for the x-axis. Default is "Time".
    ylabel (str): Label for the y-axis. Default is "Number of Species".
    title (str): Title of the plot. Default is "Number of species per abstraction over time".
    marker (str): Marker style for the plot. Default is 'o'.
    label (str): Legend label for the plot. Default is "Abstraction Size".

    Raises:
    ValueError: If the DataFrame does not contain a 'Time' column.

    Returns:
    None: Displays a line plot of abstraction sizes over time.
    """
    if 'Time' not in abstract_time_series.columns:
        # Check if the 'Time' column exists, raise error if not found
        raise ValueError("The DataFrame must include a 'Time' column for time values.")
    
    # Create a new figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Plot size 10x6 inches
    
    # Create a mapping of time and number of active species (abstraction size)
    abstraction_mapping = {
        idx: (row["Time"], len(row["Abstraction"]))  # Map index to (time, abstraction size)
        for idx, row in abstract_time_series.iterrows()
    }
    
    # Extract times and abstraction sizes
    times = [abstraction_mapping[idx][0] for idx in abstraction_mapping]  # Extract time values
    sizes = [abstraction_mapping[idx][1] for idx in abstraction_mapping]  # Extract abstraction sizes

    # Plot abstractions as ordered points
    ax.plot(times, sizes, marker=marker, linestyle='-', label=label)  # Plot with line and marker
    ax.set_xlabel(xlabel)  # Set x-axis label
    ax.set_ylabel(ylabel)  # Set y-axis label
    ax.set_title(title)  # Set plot title
    ax.grid()  # Enable grid on the plot
    ax.legend()  # Add a legend for the abstraction size
    
    plt.tight_layout()  # Adjust layout to fit all elements
    plt.show()  # Display the plot

######################################################################################## 

def plot_abstraction_sets(abstract_time_series, xlabel="Time",ylabel="Species",title="Abstraction of Time Series"):
    """
    Plots the time series abstraction generated by the abstraction_ordinary function.

    Parameters:
    abstract_time_series (pd.DataFrame): DataFrame containing 'Time' and 'Abstraction' columns.
                                         'Abstraction' should be a list of species present at each time point.
    xlabel (str): Label for the x-axis. Default is "Time".
    ylabel (str): Label for the y-axis. Default is "Presence of Species".
    title (str): Title of the plot. Default is "Abstraction of ODE Time Series".

    Returns:
    None: Displays a stacked bar chart showing the presence of species over time.
    """
    # Extract all unique species from the 'Abstraction' column
    all_species = sorted({species for row in abstract_time_series["Abstraction"] for species in row})  # Unique sorted species
    
    # Build a binary matrix indicating the presence of species over time
    binary_matrix = pd.DataFrame(
        [
            {species: (species in row) for species in all_species}  # Check presence of each species in a row
            for row in abstract_time_series["Abstraction"]
        ],
        index=abstract_time_series["Time"]  # Use time as the index
    ).astype(int)  # Convert to integer (0 or 1)
    
    # Create a new figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Plot size 10x6 inches
    
    # Plot each species as a stacked bar chart
    for i, species in enumerate(all_species):
        ax.bar(
            binary_matrix.index,  # Time points
            binary_matrix[species],  # Presence of species
            label=species,  # Label for legend
            bottom=binary_matrix.iloc[:, :i].sum(axis=1)  # Stacked position
        )
    
    # Configure plot labels and title
    ax.set_xlabel(xlabel)  # Set x-axis label
    ax.set_ylabel(ylabel)  # Set y-axis label
    ax.set_title(title)  # Set plot title
    ax.legend(title="Species")  # Add legend with title
    
    plt.show()  # Display the plot

######################################################################################
# Abstraction graph static and movie 
######################################################################################

def plot_static_abstraction_graph(abstract_time_series, title="Static Abstraction Graph"):
    """
    Plots a static abstraction graph with nodes, edges, and associated attributes.

    The graph represents:
    - Node size: Depends on the frequency of occurrences of each abstraction.
    - Node color: Represents the weighted average time of occurrence (normalized).
    - Edge thickness: Indicates the frequency of transitions between abstractions.

    Parameters:
    abstract_time_series (pd.DataFrame): A DataFrame with the following columns:
        - 'Time': Time of occurrence for each abstraction (numeric).
        - 'Abstraction': Abstraction sets represented as iterables.
    title (str): Title of the graph. Defaults to "Static Abstraction Graph".

    Returns:
    None. Displays a static plot of the abstraction graph.
    """
    abstractions = abstract_time_series["Abstraction"]  # Extracts the abstraction column
    times = abstract_time_series["Time"]  # Extracts the time column
    nodes = abstractions.apply(tuple).value_counts()  # Counts the occurrences of each abstraction

    # Compute transition frequencies
    transitions = [(tuple(abstractions[i]), tuple(abstractions[i + 1])) for i in range(len(abstractions) - 1)]  # Transitions between consecutive abstractions
    transitions_freq = pd.Series(transitions).value_counts()  # Counts the frequency of each transition

    # Create the graph
    G = nx.DiGraph()  # Creates a directed graph

    # Add nodes with attributes
    for node, freq in nodes.items():
        node_times = times[abstractions.apply(tuple) == node]  # Filters the times associated with the current node
        weighted_time = (node_times * freq).mean()  # Calculates the weighted average time
        normalized_time = (weighted_time - times.min()) / (times.max() - times.min())  # Normalizes the time
        # color = plt.cm.coolwarm(1 - normalized_time)  # Assigns a color based on the normalized time
        color = to_hex(plt.cm.coolwarm(1 - normalized_time))  # Convierte el color a formato hexadecimal
        G.add_node(node, size=freq, color=color)  # Adds the node to the graph with attributes

    # Add edges with weights
    for (source, target), weight in transitions_freq.items():
        G.add_edge(source, target, weight=weight)  # Adds edges to the graph with the corresponding weight

    # Hierarchical layout
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")  # Calculates the node positions in a hierarchical layout
    pos = {node: (x, -y) for node, (x, y) in pos.items()}  # Inverts the y-axis so the graph is drawn from bottom to top

    # Create the plot
    plt.figure(figsize=(12, 8))  # Creates a figure with a specific size

    # Draw nodes
    node_sizes = [G.nodes[node]["size"] * 100 for node in G.nodes]  # Scales the node sizes
    node_colors = [G.nodes[node]["color"] for node in G.nodes]  # Gets the node colors
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)  # Draws the nodes

    # Draw edges
    edge_weights = [G.edges[edge]["weight"] for edge in G.edges]  # Gets the edge weights
    nx.draw_networkx_edges(G, pos, width=[w / 2 for w in edge_weights], alpha=0.7)  # Draws the edges with adjusted opacity

    # Draw node labels
    # nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")  # Draws node labels 
    node_labels = {node: f"({', '.join(map(str, node))})" for node in G.nodes}  # Formatea las etiquetas
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")  # Usa las etiquetas formateadas

    # Title and display
    plt.title(title)  # Adds a title to the graph
    plt.axis("off")  # Hides the axes
    plt.show()  # Displays the graph

def plot_static_abstraction_graph_hierarchy(abstract_time_series, title="Hierarchical Abstraction Graph"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    import networkx as nx
    import pandas as pd

    abstractions = abstract_time_series["Abstraction"]
    times = abstract_time_series["Time"]
    nodes = abstractions.apply(tuple).value_counts()

    # Compute transition frequencies
    transitions = [(tuple(abstractions[i]), tuple(abstractions[i + 1])) for i in range(len(abstractions) - 1)]
    transitions_freq = pd.Series(transitions).value_counts()

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node, freq in nodes.items():
        node_times = times[abstractions.apply(tuple) == node]
        weighted_time = (node_times * freq).mean()
        normalized_time = (weighted_time - times.min()) / (times.max() - times.min())
        color = to_hex(plt.cm.coolwarm(1 - normalized_time))
        G.add_node(node, size=freq, color=color)

    # Add edges for transitions
    for (source, target), weight in transitions_freq.items():
        G.add_edge(source, target, weight=weight)

    # Add edges based on containment relationships
    node_list = list(G.nodes)
    for i, smaller in enumerate(node_list):
        for j, larger in enumerate(node_list):
            if i != j and set(smaller).issubset(set(larger)):  # Containment relationship
                if not G.has_edge(smaller, larger):  # Avoid duplicate edges
                    G.add_edge(smaller, larger, weight=1)  # Default weight

    # Compute hierarchical positions based on containment
    levels = {}
    for node in G.nodes:
        level = sum(1 for other in G.nodes if set(node).issubset(set(other)) and node != other)
        levels[node] = level

    pos = {node: (0, -level) for node, level in levels.items()}

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Draw nodes
    node_sizes = [G.nodes[node]["size"] * 100 for node in G.nodes]
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

    # Draw edges
    edge_weights = [G.edges[edge].get("weight", 1) for edge in G.edges]  # Use default weight
    nx.draw_networkx_edges(G, pos, width=[w / 2 for w in edge_weights], alpha=0.7)

    # Draw node labels
    node_labels = {node: f"({', '.join(map(str, node))})" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")

    # Title and display
    plt.title(title)
    plt.axis("off")
    plt.show()











def plot_static_abstraction_graph_html_with_hierarchy(
    abstract_time_series,
    filename="static_abstraction_graph_hierarchy.html",
    node_size_scale=100,
    edge_width_scale=0.5,
    default_color="skyblue",
    title="Static Abstraction Graph with Hierarchy"
):
    """
    Generates an HTML visualization of a static abstraction graph considering hierarchy based on subset relationships.

    Parameters:
    abstract_time_series (pd.DataFrame): DataFrame with columns:
        - 'Time': Time of occurrence for each abstraction (numeric).
        - 'Abstraction': Abstraction sets represented as iterables.
    filename (str): Name of the output HTML file. Defaults to "static_abstraction_graph_hierarchy.html".
    node_size_scale (float): Scaling factor for node sizes.
    edge_width_scale (float): Scaling factor for edge widths.
    default_color (str): Default color for nodes.
    title (str): Title of the graph. Defaults to "Static Abstraction Graph with Hierarchy".

    Returns:
    str: Filename of the generated HTML file.
    """
    # Extract data from DataFrame
    abstractions = abstract_time_series["Abstraction"]
    times = abstract_time_series["Time"]
    nodes = abstractions.apply(tuple).value_counts()

    # Compute transition frequencies
    transitions = [(tuple(abstractions[i]), tuple(abstractions[i + 1])) for i in range(len(abstractions) - 1)]
    transitions_freq = pd.Series(transitions).value_counts()

    # Create hierarchical levels based on subset relationships
    hierarchy = {}
    sorted_nodes = sorted(nodes.keys(), key=lambda x: (len(x), x))  # Sort by size of the set and lexicographically
    for node in sorted_nodes:
        hierarchy[node] = []
        for potential_parent in sorted_nodes:
            if set(node).issubset(set(potential_parent)) and node != potential_parent:
                hierarchy[node].append(potential_parent)

    # Initialize the PyVis network
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.set_options(f"""
    {{
      "edges": {{
        "smooth": false,
        "color": "gray"
      }},
      "physics": {{
        "enabled": false,
        "stabilization": {{
          "enabled": true
        }}
      }},
      "layout": {{
        "hierarchical": {{
          "enabled": true,
          "direction": "DU",
          "sortMethod": "directed"
        }}
      }}
    }}
    """)

    # Add nodes
    for node, freq in nodes.items():
        node_times = times[abstractions.apply(tuple) == node]
        weighted_time = (node_times * freq).mean()
        normalized_time = (weighted_time - times.min()) / (times.max() - times.min())
        color = to_hex(plt.cm.coolwarm(1 - normalized_time)) if not pd.isna(normalized_time) else default_color
        net.add_node(
            str(node),
            label=f"{node}",
            title=f"Freq: {freq}, Avg Time: {weighted_time:.2f}" if not pd.isna(weighted_time) else "N/A",
            color=color,
            size=freq * node_size_scale
        )

    # Add edges based on hierarchy
    for node, parents in hierarchy.items():
        for parent in parents:
            net.add_edge(str(parent), str(node), width=1, color="black", title="Subset Relationship")

    # Add transitions as additional edges
    for (source, target), weight in transitions_freq.items():
        net.add_edge(str(source), str(target), weight=weight * edge_width_scale, title=f"Freq: {weight}", color="gray")

    # Generate HTML
    net.html = net.generate_html()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(net.html)
    return filename















def plot_abstraction_graph_movie(abstract_time_series, interval=400, title="Abstraction Graph - Time"):
    """
    Creates an animated abstraction graph that evolves over time.

    The animation highlights:
    - The current node in red.
    - Past nodes gradually fading (transparent) as time progresses.
    - The last three unique abstraction sets in varying sizes and shades of blue.
    - Node size: Depends on the frequency of occurrences of each abstraction.
    - Node color: Represents the weighted average time of occurrence (normalized).
    - Edge thickness: Indicates the frequency of transitions between abstractions.

    Parameters:
    abstract_time_series (pd.DataFrame): A DataFrame with the following columns:
        - 'Time': Time of occurrence for each abstraction (numeric).
        - 'Abstraction': Abstraction sets represented as iterables.
    interval (int): Interval in milliseconds between frames of the animation. Defaults to 400.
    title (str): Title of the graph animation. Defaults to "Abstraction Graph - Time".

    Returns:
    None. Displays an animation of the abstraction graph.
    """
    abstractions = abstract_time_series["Abstraction"]
    times = abstract_time_series["Time"]
    unique_abstractions = abstractions.apply(tuple).drop_duplicates(keep='last')  # Keep last occurrences of unique abstractions
    last_three_unique = unique_abstractions.tail(3)  # Last three unique sets

    nodes = abstractions.apply(tuple).value_counts()

    # Compute transition frequencies
    transitions = [(tuple(abstractions[i]), tuple(abstractions[i + 1])) for i in range(len(abstractions) - 1)]
    transitions_freq = pd.Series(transitions).value_counts()

    # Create the graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node, freq in nodes.items():
        node_times = times[abstractions.apply(tuple) == node]
        weighted_time = (node_times * freq).mean()
        normalized_time = (weighted_time - times.min()) / (times.max() - times.min())
        color = to_hex(plt.cm.Blues(normalized_time))  # Convert the color to hexadecimal
        G.add_node(node, size=freq, color=color)

    # Add edges with weights
    for (source, target), weight in transitions_freq.items():
        G.add_edge(source, target, weight=weight)

    # Hierarchical layout
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    pos = {node: (x, -y) for node, (x, y) in pos.items()}

    # Create the figure and axis for animation
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Sizes and colors for the last three nodes
    highlight_sizes = [1 / 4, 1 / 2, 1]
    highlight_colors = ['lightblue', 'dodgerblue', 'blue']  # Different shades of blue

    # Create a function for drawing each frame
    def update_frame(i):
        ax.clear()  # Clear the current frame

        # Draw the nodes
        current_abstraction = abstractions.iloc[i]
        visited_nodes = abstractions.iloc[:i + 1].apply(tuple).value_counts().index
        node_sizes = [G.nodes[node]["size"] * 100 for node in G.nodes]

        # Adjust node colors based on visited state, with transparency for past nodes
        node_colors = [
            G.nodes[node]["color"] if node not in visited_nodes else (0.8, 0.8, 1, 0.3)  # Fade visited nodes
            for node in G.nodes
        ]

        # Highlight the last three unique nodes
        for idx, last_node in enumerate(last_three_unique):
            if last_node in G.nodes:
                node_index = list(G.nodes).index(last_node)
                node_sizes[node_index] = highlight_sizes[idx] * 500  # Update size
                node_colors[node_index] = highlight_colors[idx]  # Update color

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)

        # Draw the edges with width based on transition frequency
        edge_weights = [G.edges[edge]["weight"] for edge in G.edges]
        nx.draw_networkx_edges(G, pos, width=[w / 2 for w in edge_weights], alpha=0.7, ax=ax)

        # Draw the current node as a larger, fully opaque red node
        current_node = tuple(current_abstraction)
        nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_size=500, node_color='red', ax=ax)

        # Add labels for nodes
        node_labels = {node: f"({', '.join(map(str, node))})" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black", ax=ax)

        # Title
        ax.set_title(f"{title} {i}", fontsize=15)

        return ax,

    # Create the animation by updating the frame for each timestep
    ani = animation.FuncAnimation(fig, update_frame, frames=len(abstractions), interval=interval, blit=False, repeat=False)

    # Show the animation
    plt.show()

######################################################################################
# Histograms
######################################################################################

def plot_join_concentration_histofram(time_series, bins=30, alpha=0.7, color='skyblue', edgecolor='black', xlabel="Concentration", ylabel="Frequency", title="Histogram of Species Concentrations"):
    """
    Plots a histogram of combined species concentrations from the time series.

    Parameters:
    ----------
    time_series : pd.DataFrame
        Time series with a 'Time' column and species concentrations as additional columns.
    bins : int, optional
        Number of bins for the histogram (default is 30).
    alpha : float, optional
        Transparency level for the histogram bars (default is 0.7).
    color : str, optional
        Color of the histogram bars (default is 'skyblue').
    edgecolor : str, optional
        Color of the edges of the histogram bars (default is 'black').
    xlabel : str, optional
        Label for the x-axis (default is "Concentration").
    ylabel : str, optional
        Label for the y-axis (default is "Frequency").
    title : str, optional
        Title of the plot (default is "Histogram of Species Concentrations").
    """
    if 'Time' not in time_series.columns:
        raise ValueError("The DataFrame must include a 'Time' column for time values.")
    
    concentrations = time_series.drop(columns='Time').values.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(concentrations, bins=bins, alpha=alpha, color=color, edgecolor=edgecolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_reaction_rate_histogram(reaction_rates, bins=10, color='skyblue', edgecolor='black', alpha=0.7, xlabel="Reaction Rate", ylabel="Frequency", title="Histogram of Reaction Rates"):
    """
    Generates a histogram for the reaction rates of species.

    Parameters:
    ----------
    reaction_rates : pd.DataFrame
        DataFrame with reaction rates for species. Each column represents a species, and rows represent different time points.
    bins : int, optional
        Number of bins in the histogram (default is 10).
    color : str, optional
        Color of the histogram bars (default is 'skyblue').
    edgecolor : str, optional
        Color of the edges of the histogram bars (default is 'black').
    alpha : float, optional
        Transparency level for the histogram bars (default is 0.7).
    xlabel : str, optional
        Label for the x-axis (default is "Reaction Rate").
    ylabel : str, optional
        Label for the y-axis (default is "Frequency").
    title : str, optional
        Title of the histogram (default is "Histogram of Reaction Rates").
    """
    all_rates = reaction_rates.values.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_rates, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_species_histograms(reaction_rates, species_names, bins=10, alpha=0.7, color="skyblue", edgecolor="black", xlabel="Concentration", ylabel="Frequency", title_plot="Histogram of"):
    """
    Generates individual histograms for each species in the reaction rates DataFrame.

    Parameters:
    ----------
    reaction_rates : pd.DataFrame
        DataFrame with reaction rates for species.
    species_names : list
        List of species names (column names in the DataFrame).
    bins : int, optional
        Number of bins for the histograms (default is 10).
    alpha : float, optional
        Transparency level for the histogram bars (default is 0.7).
    color : str, optional
        Color of the histogram bars (default is "skyblue").
    edgecolor : str, optional
        Color of the edges of the histogram bars (default is "black").
    xlabel : str, optional
        Label for the x-axis (default is "Concentration").
    ylabel : str, optional
        Label for the y-axis (default is "Frequency").
    title_plot : str, optional
        Prefix for the titles of each subplot (default is "Histogram of").
    """
    num_species = len(species_names)
    fig, axes = plt.subplots(1, num_species, figsize=(5 * num_species, 4), sharey=True)

    if num_species == 1:
        axes = [axes]

    for ax, species in zip(axes, species_names):
        ax.hist(reaction_rates[species], bins=bins, alpha=alpha, color=color, edgecolor=edgecolor, label=species)
        ax.set_title(f"{title_plot} {species}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()  # Adds the legend to the subplot.

    plt.tight_layout()
    plt.show()


def plot_combined_species_histogram(reaction_rates, species_names, bins=10, alpha=0.7, edgecolor="black", xlabel="Concentration", ylabel="Frequency", title_plot="Combined Histogram"):
    """
    Generates a combined histogram of reaction rates for a set of species.

    Parameters:
    ----------
    reaction_rates : pd.DataFrame
        DataFrame with reaction rates for species.
    species_names : list
        List of species names (columns in the DataFrame).
    bins : int, optional
        Number of bins for the histograms (default is 10).
    alpha : float, optional
        Transparency level for the histogram bars (default is 0.7).
    edgecolor : str, optional
        Color of the edges of the histogram bars (default is "black").
    xlabel : str, optional
        Label for the x-axis (default is "Concentration").
    ylabel : str, optional
        Label for the y-axis (default is "Frequency").
    title_plot : str, optional
        Title of the plot (default is "Combined Histogram").
    """
    plt.figure(figsize=(8, 6))

    for species in species_names:
        plt.hist(reaction_rates[species], bins=bins, alpha=alpha, label=species, edgecolor=edgecolor)
    
    plt.title(title_plot)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Species")
    plt.grid(axis="y", linestyle="--", alpha=alpha)
    plt.tight_layout()
    plt.show()