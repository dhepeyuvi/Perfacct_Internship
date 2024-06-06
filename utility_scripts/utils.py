import matplotlib.pyplot as plt
import os
import time
import torch as trch
import pandas as pd
import nvidia_smi
import numpy as np
from skimage.transform import resize
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v3 import decode_predictions
from EMA import (
    EMA_region_begin,
    EMA_region_define,
    EMA_region_end,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

sns.set_style("darkgrid")


np.random.seed(42)


def make_preds(
    model, model_name, img, preprocessor, trt=False, torch=False, onnx=False
):
    inp_data, batching_time = batch_sigle_img(
        img,
        target_size=(224, 224),
        num_images=1,
        preprocessor=preprocessor,
    )
    top = 3
    st_time = time.time()
    if trt:
        preds = model.predict(inp_data)
        print(
            f"\n=============== Top {top} predictions by {model_name} "
            f"===============\n{decode_predictions(preds.numpy(), top=top)[0]}"
            f"\n\n"
        )
    elif torch:
        inp_data = trch.tensor(
            inp_data.transpose(0, 2, 1, 3), dtype=trch.float32
        ).cuda()
        preds = model(inp_data).detach().cpu()
        print(
            f"\n=============== Top {top} predictions by {model_name} "
            f"===============\n{decode_predictions(preds.numpy(), top=top)[0]}"
            f"\n\n"
        )
    elif onnx:
        preds = model.run(["Predictions"], {"input": inp_data})
        preds = np.squeeze(preds, axis=0)
        print(
            f"\n=============== Top {top} predictions by {model_name} "
            f"===============\n{decode_predictions(preds, top=top)[0]}\n\n"
        )
    else:
        preds = model(inp_data)
        print(
            f"\n=============== Top {top} predictions by {model_name} "
            f"===============\n{decode_predictions(preds.numpy(), top=top)[0]}"
            "\n\n"
        )
    end_time = time.time()
    pred_time = end_time - st_time
    return pred_time


def batch_sigle_img(
    img, target_size=(224, 224), num_images=4096, preprocessor=None
):
    st_time = time.time()
    img = resize(img, target_size)
    img = 255 * np.expand_dims(img, axis=0)
    if preprocessor is not None:
        img = preprocessor(img)
    inp_data = np.array(np.repeat(img, num_images, axis=0), dtype=np.float32)
    end_time = time.time()
    total_time = end_time - st_time
    print(
        f"\n================ Time to load the data onto CPU {total_time} "
        f"secs ================\n\n"
    )
    return inp_data, total_time


def generate_random_data(target_size, num_samples=4096):
    """
    Generate random data of given target size ex. 224,224

    @param _type_ target_size: IMG_W, IMG_H
    @param int num_samples: total samples to generate, defaults to 4096
    :return np.array:
    """
    # Assuming target_size is a tuple (height, width, channels)
    random_data = np.random.rand(num_samples, *target_size, 3).astype(
        np.float32
    )
    return random_data


def warm_up_model(
    model,
    input_data,
    batch_size=32,
    num_iterations=50,
    trt=False,
    onnx=False,
    torch=False,
):
    # Having a fixed batch_size for every model
    st_time = time.time()
    if trt:
        for _ in range(num_iterations):
            _ = model.predict(input_data[:batch_size])
    elif onnx:
        for _ in range(num_iterations):
            _ = model.run(["Predictions"], {"input": input_data[:batch_size]})
    elif torch:
        for _ in range(num_iterations):
            batch_input = input_data[:batch_size]
            batch_input = trch.tensor(
                batch_input.transpose(0, 2, 1, 3), dtype=trch.float32
            ).cuda()
            _ = model(batch_input)
    else:
        for _ in range(num_iterations):
            _ = model(input_data[:batch_size])
    end_time = time.time()
    total_time = end_time - st_time
    print(
        f"\n================ Batch Size {batch_size} WarmUp Time "
        f"{total_time} secs ================\n\n"
    )


def measure_performance(
    model,
    input_data,
    framework_name,
    batch_size=32,
    num_warmup_runs=50,
    num_model_runs=1,
    trt=False,
    torch=False,
    onnx=False,
    gpu_id=1,
):
    elapsed_times = []
    throughputs = []
    latencies = []
    gpu_memory_usage = []
    all_predictions = []

    nvidia_smi.nvmlInit()

    # Warm-up runs
    warm_up_model(
        model=model,
        input_data=input_data,
        batch_size=batch_size,
        num_iterations=num_warmup_runs,
        trt=trt,
        onnx=onnx,
        torch=torch,
    )

    # Actual runs
    for i in range(num_model_runs):
        # Create a measurement region. First argument is the region name.
        # It will be used in the output.
        print(f"region {os.getcwd(), os.getpid()}")
        # Code to be measured.
        region = EMA_region_define(f"{framework_name}_{batch_size}_{i}")
        # Start measurement for region0.
        EMA_region_begin(region)

        start_time = time.time()
        batch_predictions = []
        gpu_memory_per_batch = []

        for j in range(0, len(input_data), batch_size):
            batch_input = input_data[j : j + batch_size]

            # Measure GPU memory usage
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_per_batch.append(info.used)

            # print(
            #     f"Batch Size: {batch_size} and Batch "
            #     f"{j//batch_size + 1}/{len(input_data)//batch_size}:"
            #     f"GPU Memory Used: {info.used / (1024**2):.2f} MB"
            # )

            if trt:
                batch_output = model.predict(batch_input)
                batch_predictions.append(batch_output.numpy())
            elif torch:
                batch_input = trch.tensor(
                    batch_input.transpose(0, 2, 1, 3), dtype=trch.float32
                ).cuda()
                batch_output = model(batch_input)
                batch_predictions.append(batch_output.detach().cpu().numpy())
            elif onnx:
                batch_output = model.run(
                    ["Predictions"], {"input": batch_input}
                )
                batch_output = np.squeeze(batch_output, axis=0)
                batch_predictions.append(batch_output)

            else:
                batch_output = model(batch_input)
                batch_predictions.append(batch_output.numpy())

        end_time = time.time()

        # Stop measurement for region0.
        EMA_region_end(region)

        single_run_time = end_time - start_time
        throughput = len(input_data) / single_run_time
        latency = single_run_time / len(input_data)

        elapsed_times.append(single_run_time)
        throughputs.append(throughput)
        latencies.append(latency)
        gpu_memory_usage.append(np.mean(gpu_memory_per_batch))

    nvidia_smi.nvmlShutdown()
    if i == num_warmup_runs - 1:
        all_predictions.append(batch_predictions)

    return (
        elapsed_times,
        throughputs,
        latencies,
        gpu_memory_usage,
        all_predictions,
    )


def batch_model_performances(
    framework_name,
    model,
    input_data,
    batch_sizes,
    num_warmup_runs,
    num_model_runs,
    csv_path,
    trt=False,
    torch=False,
    onnx=False,
    gpu_id=1,
):

    results_list = []
    st_time = time.time()
    for batch_size in batch_sizes:
        (
            elapsed_times,
            throughputs,
            latencies,
            gpu_memory_usage,
            predictions,
        ) = measure_performance(
            model,
            input_data,
            framework_name,
            batch_size,
            num_warmup_runs,
            num_model_runs,
            trt=trt,
            torch=torch,
            onnx=onnx,
            gpu_id=gpu_id,
        )
        results_dict = {
            "Batch_Size": batch_size,
            "Elapsed_Times": elapsed_times,
            "Throughput": throughputs,
            "Latency": latencies,
            "GPU_Memory_Usage": gpu_memory_usage,
        }

        results_list.append(results_dict)
    end_time = time.time()
    total_time = end_time - st_time
    # Create separate DataFrames for each dictionary
    df_list = [pd.DataFrame(results_dict) for results_dict in results_list]

    # Concatenate the DataFrames
    results_df = pd.concat(df_list, ignore_index=True)

    # Save results to CSV
    results_df.to_csv(csv_path, index=False)
    print(
        f"\n================ Total time to run batch_performance_func "
        f"{total_time} secs ================\n"
    )
    print(
        f"\n================ Performance results saved to {csv_path} "
        f"================\n\n"
    )

    return total_time


def print_batch_predictions(all_predictions, pred_decoder, batch_number=None):
    if batch_number is not None:
        # Print predictions for a specific batch
        if 0 <= batch_number < len(all_predictions):
            batch_predictions = all_predictions[batch_number]
            print(f"\nBatch Number: {batch_number + 1}")
            for j, prediction in enumerate(batch_predictions):
                prediction_2d = np.expand_dims(prediction, axis=0)
                print(f"{j + 1}: {pred_decoder(prediction_2d, top=1)[0]}")
        else:
            print(
                f"\nInvalid batch number. Please provide a valid batch "
                f"number (1 to {len(all_predictions)})"
            )
    else:
        # Print predictions for all batches
        for i, batch_predictions in enumerate(all_predictions):
            print_batch_predictions(all_predictions, batch_number=i)


def read_csv_files_and_add_filename(folder_path):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Mean out the throughputs and latencies
            df = df.groupby("Batch_Size").mean()

            # Add a new column 'file_name' with the base name of the file

            df["framework"] = file_name.split("_")[0]
            df["uni_or_work"] = file_name.split("_")[-2]
            df["nb_or_py"] = file_name.split("_")[-1].split(".")[
                0
            ]  # Second split is to remove file_extension.
            # Append the DataFrame to the list
            dfs.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=False)

    return combined_df


def filter_data(data, uni, nb):
    df = data.copy()
    filtered_data = df[
        (
            ((df["uni_or_work"] == "uni") & (df["nb_or_py"] == "nb"))
            if (uni and nb)
            else (
                ((df["uni_or_work"] == "uni") & (df["nb_or_py"] == "py"))
                if (uni and not nb)
                else (
                    ((df["uni_or_work"] == "work") & (df["nb_or_py"] == "nb"))
                    if (not uni and nb)
                    else (
                        (
                            (df["uni_or_work"] == "work")
                            & (df["nb_or_py"] == "py")
                        )
                        if (not uni and not nb)
                        else False
                    )
                )
            )
        )
    ]
    return filtered_data


def generate_benchmark_plot(
    model_name,
    csv_folder_path,
    save_path,
    uni,
    nb,
    benchmark_col_name="Throughput",
    v100=False,
    invert=True,
):
    if v100:
        title_prefix = "Uni V100" if uni else "Work V100"
    else:
        title_prefix = "Uni" if uni else "Work"

    title_prefix += " Jnb" if nb else " Py"
    # Specify the folder path where the CSV files are located
    folder_path = csv_folder_path
    # Call the function to read CSV files, add filename column, and
    # combine into a single DataFrame
    result_df = read_csv_files_and_add_filename(folder_path)
    result_df = filter_data(result_df, uni, nb)
    # Define the desired order
    if v100:
        # Remove the first 4 letters i.e. V100 from the file_name to
        # have normal name of frameworks
        result_df["framework"] = result_df["framework"].str[4:]

    result_df["framework"] = result_df["framework"].str.capitalize()
    order = [
        "Tf",
        "Tfxla",
        "Keras",
        "Kerasxla",
        "Tftrtfp16",
        "Tftrtfp32",
        "Onnxrt",
        "Torch",
    ]

    # Sort the DataFrame based on the 'file_name' column order
    result_df["framework"] = pd.Categorical(
        result_df["framework"], categories=order, ordered=True
    )
    result_df = result_df.sort_values("framework")

    # Optionally, reset the index if needed
    result_df = result_df.reset_index()
    result_df = result_df.reset_index(drop=True)

    file_name_colors = {
        frmwrk: sns.color_palette("rainbow", n_colors=len(order))[i]
        for i, frmwrk in enumerate(order)
    }
    file_name_colors = dict(
        sorted(file_name_colors.items(), key=lambda item: order.index(item[0]))
    )

    # Scale the benchmark column by 1000x if it is latency
    if benchmark_col_name.lower() == "latency":
        result_df[benchmark_col_name] *= 1000  # Scaling by 1000x
        y_label = "Latency (ms)"
        y_axis_lim = 20
        yticks = range(0, y_axis_lim, 5)

    else:
        y_label = benchmark_col_name  # Use the original label
        y_axis_lim = 3200
        yticks = range(0, y_axis_lim, 200)

    # Create a bar plot using Seaborn
    g = sns.catplot(
        data=result_df,
        kind="bar",
        x="Batch_Size",
        y=benchmark_col_name,
        hue="framework",
        orient="v",
        legend_out=True,
        aspect=3,
        height=6,
        palette=file_name_colors.values(),
        # Use a predefined color palette
    )
    # Annotate the values inside the bars, parallel to them
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() != 0:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height() / 1.8),
                    ha="center",
                    va="center",
                    rotation="vertical",
                    fontsize=10,
                    color="black",
                )  # Adjust color and other parameters as needed

    # Additional styling
    g.despine(left=True)

    g.set_axis_labels("Batch_Size", y_label, fontsize=14)
    g.legend.set_title("Frameworks")
    g.set(
        ylim=y_axis_lim,
        yticks=yticks,
    )

    # NOTE: Add this only if the plots are inverted, restart the
    # jupyter nb, if this part is un-commented
    # Invert the y-axis
    if invert:
        g.ax.invert_yaxis()

    title = f"{title_prefix} {model_name} {y_label} "
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Show the plot

    plt.savefig(
        f"{save_path}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()


def process_csv(df, gpu_ids):
    # Split 'region_idf' column to extract framework, batch_size, and iteration
    df[["framework", "batch_size", "iteration"]] = df["region_idf"].str.split(
        "_", expand=True
    )

    # Convert 'iteration' and 'batch_size' columns to numeric
    df["iteration"] = pd.to_numeric(df["iteration"])
    try:
        df["batch_size"] = pd.to_numeric(df["batch_size"])
    except Exception as ex:
        print(f"Error {ex} for {df.head()}")

    # Drop unnecessary columns
    df.drop(
        ["file", "line", "function", "visits", "region_idf"],
        axis=1,
        inplace=True,
    )

    # Initialize list to store selected rows
    selected_rows = []

    # Iterate over each framework and select the corresponding row
    # indices based on gpu_ids
    for key, frameworks in gpu_ids.items():
        for framework in frameworks:
            # Filter rows for the current framework
            framework_rows = df[df["framework"] == framework]
            # Select the nth row for each group defined by
            # ['framework', 'batch_size', 'iteration']
            nth_rows = framework_rows.groupby(
                ["framework", "batch_size", "iteration"]
            ).nth(key)
            # Append selected rows to the list
            selected_rows.append(nth_rows)

    # Concatenate the selected rows into a single DataFrame
    selected_df = pd.concat(selected_rows)
    df_avg = selected_df.groupby(["framework", "batch_size"]).agg(
        {"energy": "mean", "time": "mean"}
    )

    df_avg["power_watts"] = df_avg["energy"] / df_avg["time"]

    # return df_avg
    df_avg = df_avg.reset_index()
    df_avg = df_avg.reset_index(drop=True)
    return df_avg


def EMA_Process_folder(folder_path, gpu_ids, uni=True, nb=False, v100=False):
    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter files based on the terminal_run argument
    if nb:
        files = [f for f in files if f.endswith("_nb.csv")]
    else:
        files = [f for f in files if f.endswith("_py.csv")]

    # Process each file and store the average metrics DataFrame in a list
    dfs = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        if not uni:
            # Remove the rows which
            df = df[~df["device_name"].str.startswith("CPU")]
        df_metrics = process_csv(df, gpu_ids)
        dfs.append(df_metrics)

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs)

    if v100:
        suffix = "v100_uni" if uni else "v100_work"
    else:
        suffix = "uni" if uni else "work"
    suffix += "_jnb" if nb else "_py"
    # Define the new CSV file name
    new_csv_name = f"EMA_{suffix}_final.csv"

    # Save the concatenated DataFrame to a new CSV file
    output_csv_path = os.path.join(folder_path, new_csv_name)
    concatenated_df["framework"] = concatenated_df[
        "framework"
    ].str.capitalize()

    order = [
        "Tf",
        "Tfxla",
        "Keras",
        "Kerasxla",
        "Tftrtfp16",
        "Tftrtfp32",
        "Onnxrt",
        "Torch",
    ]

    # Sort the DataFrame based on the 'file_name' column order
    concatenated_df["framework"] = pd.Categorical(
        concatenated_df["framework"], categories=order, ordered=True
    )
    concatenated_df = concatenated_df.sort_values("framework")
    concatenated_df = concatenated_df.sort_values(
        by=["framework", "batch_size"]
    )
    concatenated_df.reset_index(drop=True, inplace=True)
    concatenated_df.to_csv(output_csv_path, index=False)

    return concatenated_df


def EMA_Plotter(
    model_name,
    gpu_ids,
    csv_folder_path,
    save_path,
    uni,
    v100=False,
    benchmark_col_name="energy",
    nb=False,
    invert=True,
):
    if v100:
        title_prefix = "Uni V100" if uni else "Work V100"
    else:
        title_prefix = "Uni" if uni else "Work"
    title_prefix += " Jnb" if nb else " Py"

    df = EMA_Process_folder(
        csv_folder_path, gpu_ids=gpu_ids, uni=uni, nb=nb, v100=v100
    )
    ppath = Path(csv_folder_path).resolve().parents[0]
    benchmark_df = ordered_results(ppath, uni, nb, v100)

    mdf = pd.merge(
        benchmark_df,
        df,
        left_on=["framework", "Batch_Size"],
        right_on=["framework", "batch_size"],
        suffixes=("_df1", "_df2"),
    )
    mdf["energy_eff"] = (mdf["Throughput"] / (mdf["energy"])) * 1e6
    mdf["time_eff"] = (mdf["Throughput"] / (mdf["time"])) * 1e6
    mdf["power_eff"] = mdf["Throughput"] / (mdf["power_watts"])

    df = mdf
    order = [
        "Tf",
        "Tfxla",
        "Keras",
        "Kerasxla",
        "Tftrtfp16",
        "Tftrtfp32",
        "Onnxrt",
        "Torch",
    ]
    frmwrk_colors = {
        frmwrk: sns.color_palette("rainbow", n_colors=len(order))[i]
        for i, frmwrk in enumerate(order)
    }

    # Scale the benchmark column by 1000x if it is latency
    if benchmark_col_name.lower() == "energy":
        df[benchmark_col_name] /= 1000000  # Energy is in micro Joules
        y_axis_lim = 600
        yticks = range(0, y_axis_lim, 50)
        y_label = "Energy (joules)"
    elif benchmark_col_name.lower() == "time":
        df[benchmark_col_name] /= 1000000  # Time is in micro seconds
        y_label = "Time (secs)"
        y_axis_lim = 10
        yticks = range(0, y_axis_lim, 2)
    elif benchmark_col_name.lower() == "energy_eff":
        y_label = "Energy Eff (#preds per sec per Joule)"
        y_axis_lim = 300
        yticks = range(0, y_axis_lim, 50)
    elif benchmark_col_name.lower() == "power_eff":
        y_label = "Power Eff (#preds per sec per watt)"
        y_axis_lim = 50
        yticks = range(0, y_axis_lim, 10)
    else:
        y_label = "Power (watts)"  # Use the original label
        y_axis_lim = 300
        yticks = range(0, y_axis_lim, 50)

    # Create a bar plot using Seaborn
    g = sns.catplot(
        data=df,
        kind="bar",
        x="batch_size",
        y=benchmark_col_name,
        hue="framework",
        orient="v",
        legend_out=True,
        aspect=3,
        height=6,
        palette=frmwrk_colors.values(),
        # Use a predefined color palette
    )

    # Annotate the values inside the bars, parallel to them
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() != 0:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height() / 1.8),
                    ha="center",
                    va="center",
                    rotation="vertical",
                    fontsize=10,
                    color="black",
                )  # Adjust color and other parameters as needed

    g.despine(left=True)
    g.set_axis_labels("Batch_Size", y_label, fontsize=14)
    g.legend.set_title("Frameworks")
    g.set(ylim=y_axis_lim, yticks=yticks)

    if invert:
        g.ax.invert_yaxis()  # Necessary to keep y_axis from 0 to top

    title = f"EMA {title_prefix} {model_name} {y_label} "
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Show the plot

    plt.savefig(
        f"{save_path}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()


def ordered_results(folder_path, uni, nb, v100=False):
    result_df = read_csv_files_and_add_filename(folder_path)
    result_df = filter_data(result_df, uni, nb)
    # Define the desired order
    if v100:
        # Remove the first 4 letters i.e. V100 from the file_name to
        # have normal name of frameworks
        result_df["framework"] = result_df["framework"].str[4:]

    result_df["framework"] = result_df["framework"].str.capitalize()
    order = [
        "Tf",
        "Tfxla",
        "Keras",
        "Kerasxla",
        "Tftrtfp16",
        "Tftrtfp32",
        "Onnxrt",
        "Torch",
    ]

    # Sort the DataFrame based on the 'framework' column order
    result_df["framework"] = pd.Categorical(
        result_df["framework"], categories=order, ordered=True
    )
    result_df = result_df.sort_values("framework")
    return result_df


def concatenate_thrpt_latency_csv(column_name, dataframe_gpu_dict, save_path):
    # Add a new index column to each dataframe and reset index
    for df_name, df in dataframe_gpu_dict.items():
        df["gpu_name"] = df_name
        df["Batch_Size"] = df.index
        # df.reset_index(drop=True, inplace=True)s

    # Concatenate the dataframes representing different GPUs
    concatenated_df = pd.concat(dataframe_gpu_dict.values(), axis=0)
    # concatenated_df.reset_index(drop=True, inplace=True)

    # Iterate over the unique values in the specified column
    unique_values = concatenated_df["framework"].unique()

    pivot_dfs = []
    for value in unique_values:
        # Filter the concatenated dataframe for each value
        value_df = concatenated_df[concatenated_df["framework"] == value]

        # Pivot the dataframe to have GPU names as columns and batch sizes
        # as rows
        pivot_df = value_df.pivot(
            index="Batch_Size", columns="gpu_name", values=[column_name]
        )

        # Add a new column for the specified column name
        pivot_df["framework"] = value

        # Append the pivot dataframe to the list
        pivot_dfs.append(pivot_df)

    # Concatenate all pivot dataframes
    final_df = pd.concat(pivot_dfs)

    # Save the concatenated dataframe to a CSV file
    final_df.to_csv(save_path)


def concatenate_cols(df, column_name, save_path):
    df = df.copy()
    if column_name == "energy":
        df["energy"] = df["energy"] / 1000000
    # Iterate over the unique values in the specified column
    unique_values = df["framework"].unique()

    pivot_dfs = []
    for value in unique_values:
        # Filter the concatenated dataframe for each value
        value_df = df[df["framework"] == value]

        # Pivot the dataframe to have GPU names as columns and batch sizes
        # as rows
        pivot_df = value_df.pivot(
            index="batch_size", columns="gpu_name", values=[column_name]
        )

        # Add a new column for the specified column name
        pivot_df["framework"] = value

        # Append the pivot dataframe to the list
        pivot_dfs.append(pivot_df)

    # Concatenate all pivot dataframes
    final_df = pd.concat(pivot_dfs)

    # Save the concatenated dataframe to a CSV file
    final_df.to_csv(save_path)


def merge_benchmarks_ema_files(
    ema_csv_path, benchmark_csv_folder_path, uni, nb, v100, gpu_name
):
    ema_df = pd.read_csv(ema_csv_path)
    benchmark_df = ordered_results(benchmark_csv_folder_path, uni, nb, v100)
    mdf = pd.merge(
        benchmark_df,
        ema_df,
        left_on=["framework", "Batch_Size"],
        right_on=["framework", "batch_size"],
        suffixes=("_df1", "_df2"),
    )
    mdf["energy_eff"] = (mdf["Throughput"] / (mdf["energy"])) * 1e6
    mdf["time_eff"] = (mdf["Throughput"] / (mdf["time"])) * 1e6
    mdf["power_eff"] = mdf["Throughput"] / (mdf["power_watts"])
    mdf["gpu_name"] = gpu_name
    return mdf.copy()


def gpu_plotter(df, benchmark_col_name, model_name, save_folder):
    # Define benchmark column name mappings
    benchmark_mappings = {
        "energy": ("Energy (joules)", 1000000, 600, range(0, 600, 50)),
        "energy_eff": (
            "Energy Eff (#preds per sec per Joule)",
            1,
            300,
            range(0, 300, 50),
        ),
        "time": ("Time (secs)", 1000000, 10, range(0, 10, 2)),
        "time_eff": (
            "Time Eff calculated as per EMA",
            1,
            26000,
            range(0, 26000, 1000),
        ),
        "power_watts": ("Power (watts)", 1, 300, range(0, 300, 50)),
        "power_eff": (
            "Power Eff (#preds per sec per watt)",
            1,
            50,
            range(0, 50, 10),
        ),
        "latency": ("Latency (ms)", 1, 20, range(0, 20, 5)),
        "throughput": (
            "Throughput (#preds per sec)",
            1,
            3200,
            range(0, 3200, 200),
        ),
        "ema_energy_x_time_joulsec": (
            "Energy X Time (joules times sec)",
            1,
            3500,
            range(0, 3500, 250),
        ),
    }

    if benchmark_col_name.lower() not in benchmark_mappings:
        print(f"Incorrect {benchmark_col_name}, is not present in dataframe")
        return

    # Unpack benchmark column name mappings
    y_label, scale_factor, y_axis_lim, yticks = benchmark_mappings[
        benchmark_col_name.lower()
    ]

    # Scale the benchmark column if necessary
    if scale_factor != 1:
        df[benchmark_col_name] /= scale_factor

    # Scale latency if needed
    if benchmark_col_name.lower() == "latency":
        df[benchmark_col_name] *= 1000  # Scaling by 1000x
        benchmark_col_name = "Latency"

    # Generate unique colors for each GPU name
    order = sorted(df["gpu_name"].unique())
    gpu_colors = {
        gpu: sns.color_palette("rainbow", n_colors=len(order))[i]
        for i, gpu in enumerate(order)
    }

    # Create separate plots for each framework
    for framework, data in df.groupby("framework"):
        g = sns.catplot(
            data=data,
            kind="bar",
            x="batch_size",
            y=benchmark_col_name,
            hue="gpu_name",
            orient="v",
            legend_out=True,
            aspect=3,
            height=6,
            palette=gpu_colors,
        )

        # Annotate the values inside the bars
        for ax in g.axes.flat:
            for p in ax.patches:
                if p.get_height() != 0:
                    ax.annotate(
                        f"{p.get_height():.2f}",
                        (
                            p.get_x() + p.get_width() / 2.0,
                            p.get_height() / 1.8,
                        ),
                        ha="center",
                        va="center",
                        rotation="vertical",
                        fontsize=10,
                        color="black",
                    )

        g.despine(left=True)
        g.set_axis_labels("Batch_Size", y_label, fontsize=14)
        g.legend.set_title("GPU Names")
        g.set(ylim=(0, y_axis_lim), yticks=yticks)

        title = f"{model_name} {framework} {y_label} "
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot
        save_path = os.path.join(
            save_folder, f"{model_name}_{framework}_{benchmark_col_name}.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def ema_power_vs_thrpt_plot(csv_path, save_dir):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Sort the DataFrame by GPU name and throughput
    df = df.sort_values(by=["gpu_name", "Throughput"])

    # Define colors for frameworks
    order = [
        "Tf",
        "Tfxla",
        "Keras",
        "Kerasxla",
        "Tftrtfp16",
        "Tftrtfp32",
        "Onnxrt",
        "Torch",
    ]
    framework_colors = {
        frmwrk: sns.color_palette("rainbow", n_colors=len(order))[i]
        for i, frmwrk in enumerate(order)
    }

    # Define markers for batch sizes
    batch_sizes = sorted(df["batch_size"].unique())
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "+",
        "x",
        "|",
        "_",
    ]
    marker_map = {
        batch_size: markers[i % len(markers)]
        for i, batch_size in enumerate(batch_sizes)
    }

    # Iterate over each GPU
    for gpu, df_gpu in df.groupby("gpu_name"):
        # Create a new figure
        plt.figure(figsize=(12, 8))
        # Initialize legend handles and labels for batch sizes and frameworks
        legend_handles_batch_size = []
        legend_labels_batch_size = []
        legend_handles_framework = []
        legend_labels_framework = []
        # Iterate over each framework for the current GPU
        for i, framework in enumerate(order):
            if framework in df_gpu["framework"].unique():
                df_framework = df_gpu[df_gpu["framework"] == framework]

                # Fit linear regression
                model = LinearRegression()
                model.fit(
                    df_framework["Throughput"].values.reshape(-1, 1),
                    df_framework["power_watts"].values,
                )
                y_pred = model.predict(
                    df_framework["Throughput"].values.reshape(-1, 1)
                )
                # Calculate R-squared value
                r2 = model.score(
                    df_framework["Throughput"].values.reshape(-1, 1),
                    df_framework["power_watts"].values,
                )

                # Plot regression line
                plt.plot(
                    df_framework["Throughput"],
                    y_pred,
                    linestyle="-",
                    color=framework_colors[framework],
                    label=f"{framework} (R² = {r2:.2f})",
                )
                # Plot line plot with markers for batch sizes
                for j, batch_size in enumerate(
                    sorted(df_framework["batch_size"].unique())
                ):
                    x_batch = df_framework[
                        df_framework["batch_size"] == batch_size
                    ]["Throughput"].values
                    y_batch = df_framework[
                        df_framework["batch_size"] == batch_size
                    ]["power_watts"].values
                    plt.plot(
                        x_batch,
                        y_batch,
                        marker=marker_map[batch_size],
                        linestyle="-",
                        color=framework_colors[framework],
                    )

                    # Add legend handles and labels for batch sizes
                    if (
                        i == 0
                    ):  # Add batch size legend handles and labels for the first framework only
                        legend_handles_batch_size.append(
                            plt.Line2D(
                                [0],
                                [0],
                                linestyle="-",
                                color="black",
                                marker=marker_map[batch_size],
                                markersize=8,
                            )
                        )
                        legend_labels_batch_size.append(
                            f"Batch Size {batch_size}"
                        )

                # Add legend handles and labels for frameworks
                legend_handles_framework.append(
                    plt.Line2D(
                        [0],
                        [0],
                        linestyle="solid",
                        color=framework_colors[framework],
                    )
                )
                legend_labels_framework.append(f"{framework} (R² = {r2:.2f})")

        # Set title and labels
        plt.title(f"{gpu} Power vs Throughput")
        plt.xlabel("Throughput")
        plt.ylabel("Power (Watts)")

        # Combine legend handles and labels
        all_handles = legend_handles_batch_size + legend_handles_framework
        all_labels = legend_labels_batch_size + legend_labels_framework

        # Show legend
        plt.legend(
            all_handles,
            all_labels,
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        # Save the plot
        save_path = os.path.join(save_dir, f"{gpu}_pwr_thrpt_plot.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()


def image_per_metric_vs_batch_size_plot(
    csv_path,
    metric_column,
    framework_images=None,
    figsize=(12, 8),
    title=None,
    xlabel=None,
    ylabel=None,
    legend_title=None,
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.05, 1),
    save_dir=None,
    grid=True,
):
    data = pd.read_csv(csv_path, index_col=0)
    # Calculate the number of images per metric
    if framework_images is None:
        framework_images = {}
    for framework, num_images in framework_images.items():
        data.loc[data["framework"] == framework, metric_column] = (
            num_images
            / data.loc[data["framework"] == framework, metric_column]
        )

    # Define the order and colors for frameworks
    order = list(framework_images.keys())
    frmwrk_colors = {
        frmwrk: sns.color_palette("rainbow", n_colors=len(order))[i]
        for i, frmwrk in enumerate(order)
    }

    # Create a mapping from batch size to markers
    batch_sizes = sorted(data["batch_size"].unique())
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "+",
        "x",
        "|",
        "_",
    ]
    marker_map = {
        batch_size: markers[i % len(markers)]
        for i, batch_size in enumerate(batch_sizes)
    }

    # Plot the results
    gpu_names = data["gpu_name"].unique()
    for gpu_name in gpu_names:
        plt.figure(figsize=figsize)
        gpu_data = data[data["gpu_name"] == gpu_name]

        for framework in order:
            if framework in gpu_data["framework"].unique():
                group_data = gpu_data[gpu_data["framework"] == framework]
                group_data = group_data.sort_values(
                    by="batch_size"
                )  # Sort by batch size
                color = frmwrk_colors[framework]
                plt.plot(
                    group_data["batch_size"],
                    group_data[metric_column],
                    color=color,
                    label=framework,
                )
                for batch_size in sorted(group_data["batch_size"].unique()):
                    batch_data = group_data[
                        group_data["batch_size"] == batch_size
                    ]
                    marker = marker_map[batch_size]
                    plt.scatter(
                        batch_data["batch_size"],
                        batch_data[metric_column],
                        color=color,
                        marker=marker,
                        edgecolor="black",
                        s=100,
                    )

        # Create custom legend for batch sizes and frameworks
        handles = []
        labels = []

        for fw in order:
            if fw in gpu_data["framework"].unique():
                handles.append(
                    plt.Line2D(
                        [0], [0], color=frmwrk_colors[fw], lw=2, label=fw
                    )
                )
                labels.append(fw)
        for bs in batch_sizes:
            if bs in gpu_data["batch_size"].unique():
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker=marker_map[bs],
                        color="w",
                        markerfacecolor="k",
                        markersize=10,
                        label=f"Batch {bs}",
                    )
                )
                labels.append(f"Batch {bs}")

        plt.legend(
            handles=handles,
            labels=labels,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{gpu_name} {title}")
        plt.grid(grid)
        save_path = os.path.join(
            save_dir,
            f"{gpu_name}_images_per_{metric_column}.png",
        )
        # Show the plot
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()
