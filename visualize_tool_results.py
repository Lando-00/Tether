"""
visualize_tool_results.py - Visualize tool calling test results

This script visualizes the results of tool calling tests, showing performance
across different models or test runs.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from rich.console import Console

console = Console()

def load_result_files(directory: str = ".") -> List[Dict[str, Any]]:
    """Load all tool calling result files from the specified directory."""
    result_files = []
    for filename in os.listdir(directory):
        if filename.startswith("tool_calling_results_") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    result_files.append(data)
                    console.print(f"[green]Loaded:[/green] {filename}")
            except (json.JSONDecodeError, IOError) as e:
                console.print(f"[yellow]Error loading {filename}:[/yellow] {e}")
    
    return result_files

def visualize_success_rates(results: List[Dict[str, Any]]) -> None:
    """Visualize success rates across different models or test runs."""
    if not results:
        console.print("[yellow]No result files found to visualize[/yellow]")
        return
    
    # Sort results by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))
    
    # Extract model names and success rates
    model_names = [r.get("model_name", "Unknown") for r in results]
    success_rates = [r.get("summary", {}).get("success_rate", 0) for r in results]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(model_names)), success_rates, color='skyblue')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title('Tool Calling Success Rate by Model')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{success_rates[i]:.1f}%", ha='center')
    
    plt.ylim(0, 110)  # Set y-axis to go from 0 to 110% to accommodate the labels
    
    # Save the plot
    plt.savefig("tool_calling_success_rates.png")
    console.print("[green]Success rates visualization saved to:[/green] tool_calling_success_rates.png")
    plt.close()

def visualize_execution_times(results: List[Dict[str, Any]]) -> None:
    """Visualize execution times across different models or test runs."""
    if not results:
        return
    
    # Sort results by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))
    
    # Extract model names and average execution times
    model_names = [r.get("model_name", "Unknown") for r in results]
    avg_times = [r.get("summary", {}).get("avg_time", 0) for r in results]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(model_names)), avg_times, color='lightgreen')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Average Tool Calling Execution Time by Model')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"{avg_times[i]:.2f}s", ha='center')
    
    # Save the plot
    plt.savefig("tool_calling_execution_times.png")
    console.print("[green]Execution times visualization saved to:[/green] tool_calling_execution_times.png")
    plt.close()

def visualize_tool_performance(results: List[Dict[str, Any]]) -> None:
    """Visualize performance for different tools across test runs."""
    if not results:
        return
    
    # Collect all unique tool names across all test cases
    all_tools = set()
    for result in results:
        for test_case in result.get("test_cases", []):
            all_tools.add(test_case.get("expected_tool", "unknown"))
    
    all_tools = list(all_tools)
    
    # Calculate success rate for each tool across all results
    tool_success_rates = {tool: [] for tool in all_tools}
    model_names = []
    
    for result in results:
        model_name = result.get("model_name", "Unknown")
        model_names.append(model_name)
        
        # Initialize success count for each tool
        tool_successes = {tool: {"success": 0, "total": 0} for tool in all_tools}
        
        # Count successes for each tool
        for test_case in result.get("test_cases", []):
            tool = test_case.get("expected_tool", "unknown")
            if tool in tool_successes:
                tool_successes[tool]["total"] += 1
                if test_case.get("success", False):
                    tool_successes[tool]["success"] += 1
        
        # Calculate success rates
        for tool in all_tools:
            if tool_successes[tool]["total"] > 0:
                success_rate = (tool_successes[tool]["success"] / tool_successes[tool]["total"]) * 100
            else:
                success_rate = 0
            tool_success_rates[tool].append(success_rate)
    
    # Create a grouped bar chart
    plt.figure(figsize=(12, 7))
    x = np.arange(len(model_names))
    width = 0.8 / len(all_tools)
    
    for i, tool in enumerate(all_tools):
        offset = (i - len(all_tools)/2 + 0.5) * width
        plt.bar(x + offset, tool_success_rates[tool], width, label=tool)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title('Tool Calling Success Rate by Tool and Model')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend(title='Tools')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("tool_performance_by_model.png")
    console.print("[green]Tool performance visualization saved to:[/green] tool_performance_by_model.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize tool calling test results")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing result files")
    args = parser.parse_args()
    
    console.print("[bold]Tool Calling Results Visualization[/bold]")
    
    # Load all result files
    results = load_result_files(args.dir)
    
    if not results:
        console.print("[yellow]No result files found in the specified directory[/yellow]")
        return
    
    console.print(f"[green]Found {len(results)} result files[/green]")
    
    # Generate visualizations
    visualize_success_rates(results)
    visualize_execution_times(results)
    visualize_tool_performance(results)
    
    console.print("[bold green]Visualizations complete![/bold green]")

if __name__ == "__main__":
    main()
