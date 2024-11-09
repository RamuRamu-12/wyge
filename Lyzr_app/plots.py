import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any, Optional
from vyzeai.tools.base_tool import Tool, add_function
from pydantic import BaseModel, Field


def generate_bar_chart(x_values: List[str], y_values: List[float], title: str = "Bar Chart",
                       file_path: str = "bar_chart.png"):
    plt.figure(figsize=(8, 6))
    plt.bar(x_values, y_values)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.savefig(file_path, format='png')
    plt.close()

    return file_path


@add_function(generate_bar_chart)
class BarChartInput(BaseModel):
    """Tool to make bar chart"""
    x_values: List[str] = Field(description="Labels for the X-axis of the bar chart.")
    y_values: List[float] = Field(description="Values for the Y-axis corresponding to each X-axis label.")
    title: str = Field(default="Bar Chart", description="Title of the bar chart.")


def generate_line_chart(x_values: List[float], y_values: List[float], title: str = "Line Chart",
                        file_path: str = 'line_chart.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.savefig(file_path, format='png')
    plt.close()

    return file_path


@add_function(generate_line_chart)
class LineChartInput(BaseModel):
    """Tool to make line chart"""
    x_values: List[float] = Field(description="Values for the X-axis.")
    y_values: List[float] = Field(description="Values for the Y-axis corresponding to each X-axis value.")
    title: str = Field(default="Line Chart", description="Title of the line chart.")


def generate_scatter_plot(x_values: List[float], y_values: List[float], title: str = "Scatter Plot"):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64


class ScatterPlotInput(BaseModel):
    """Tool to make scatter plot"""
    x_values: List[float] = Field(description="Values for the X-axis.")
    y_values: List[float] = Field(description="Values for the Y-axis corresponding to each X-axis value.")
    title: str = Field(default="Scatter Plot", description="Title of the scatter plot.")


def generate_pie_chart(labels: List[str], sizes: List[float], title: str = "Pie Chart"):
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64


class PieChartInput(BaseModel):
    """Tool to make pie chart"""
    labels: List[str] = Field(description="Labels for each section of the pie chart.")
    sizes: List[float] = Field(description="Sizes of each section of the pie chart, typically in percentages.")
    title: str = Field(default="Pie Chart", description="Title of the pie chart.")


def generate_histogram(data: List[float], bins: int = 10, title: str = "Histogram"):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64


class HistogramInput(BaseModel):
    """Tool to make histogram"""
    data: List[float] = Field(description="The data points to be plotted in the histogram.")
    bins: int = Field(default=10, description="Number of bins in the histogram.")
    title: str = Field(default="Histogram", description="Title of the histogram.")


def generate_box_plot(data: List[List[float]], labels: List[str], title: str = "Box Plot"):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Values')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64


class BoxPlotInput(BaseModel):
    """Tool to make box plot"""
    data: List[List[float]] = Field(
        description="A list of datasets to be plotted, where each dataset is a list of values.")
    labels: List[str] = Field(description="Labels for each dataset on the X-axis.")
    title: str = Field(default="Box Plot", description="Title of the box plot.")


# def execute_python_code(code: str, global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     try:
#         if global_vars is None:
#             global_vars = {}

#         local_vars = {}
#         exec(code, global_vars, local_vars)

#         # print(local_vars)
#         serializable_vars = {k: v for k, v in local_vars.items() if not callable(v) and not isinstance(v, type)}

#         return {"result": str(serializable_vars), "error": None}
#     except Exception as e:
#         return {"result": {}, "error": str(e)}
#         # raise e

def execute_code_parts(code_parts: List[str], global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes a list of code parts sequentially, retaining state between executions.

    Args:
        code_parts (List[str]): A list of Python code parts to be executed in sequence.
        global_vars (Optional[Dict[str, Any]]): A dictionary of global variables to use during code execution.

    Returns:
        Dict[str, Any]: A dictionary containing either the local variables from the executed code or an error message.
    """
    if global_vars is None:
        global_vars = {}

    local_vars = {}

    # Iterate through each code part and execute it
    for idx, code_part in enumerate(code_parts):
        try:
            exec(code_part, global_vars, local_vars)
            global_vars.update(local_vars)
            print(f"Part {idx + 1} executed successfully.")
        except Exception as e:
            print(f"Error in part {idx + 1}: {e}")
            return {"result": {}, "error": str(e)}


    print("Returning...........")
    # Return local vars for inspection after all parts are executed
    return {"result": str(local_vars), "error": None}


@add_function(execute_code_parts)
class PythonCodeExecution(BaseModel):
    """
    Executes a list of Python code parts sequentially, with each execution being independent and isolated.

    **Note:** Variables and states do not persist between executions, including global variables and local variables.
    Any imports or global variables provided in `global_vars` must be redefined for each execution.
    """
    code_parts: List[str] = Field(..., description="A list of Python code parts to be executed in sequence.")
    global_vars: Optional[Dict[str, Any]] = Field(
        default=None,
        description="imports or Global variables to use during execution. Example: {'math': __import__('math')}"
    )


# bar_plot_tool = Tool(BarChartInput)()
# line_plot_tool = Tool(LineChartInput)()

# tools = [bar_plot_tool, line_plot_tool]

def install_library(library_name: str) -> None:
    """
    Installs the required library using pip.

    Args:
        library_name (str): Name of the library to install.
    """
    import subprocess
    import sys
    print(f"Attempting to install missing library: {library_name}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    except Exception as e:
        return f"Unable to install {library_name}. Error: {e}"

    return f"{library_name} has been installed sucessfully."


@add_function(install_library)
class InstallLibrary(BaseModel):
    """
    Installs the required library using pip.
    """
    library_name: str = Field(description="Name of the library to install.")


lib_tool = Tool(InstallLibrary)()
py_tool = Tool(PythonCodeExecution)()
tools = [py_tool, lib_tool]