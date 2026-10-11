import torch
import numpy as np
import subprocess
import tempfile
import os
import sys
from dhsic import dhsic


def check_r_dhsic_installed():
    """Check if dHSIC package is installed in R."""
    r_script = """
    # Set library path to user's R library
    user_lib <- normalizePath(path.expand("~/R/library"), winslash = "/", mustWork = FALSE)
    .libPaths(c(user_lib, .libPaths()))

    if (!require("dHSIC")) {
        cat("NOT_INSTALLED")
        quit(status=1)
    }
    cat("INSTALLED")
    """

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False) as f:
        f.write(r_script.encode())
        r_script_file = f.name

    try:
        result = subprocess.run(
            ["Rscript", r_script_file], capture_output=True, text=True
        )
        return "INSTALLED" in result.stdout
    except subprocess.CalledProcessError:
        return False
    finally:
        os.unlink(r_script_file)


def install_r_dhsic():
    """Install dHSIC package in R."""
    print("\nInstalling dHSIC package in R...")

    try:
        # Use the separate R script file
        result = subprocess.run(
            ["Rscript", "--vanilla", "install_dhsic.R"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Error installing dHSIC package:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Error running R script: {str(e)}")
        return False


def generate_test_data(n_samples, seed=0):
    """Generate test data with fixed random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Case 1: Three independent Gaussian variables
    x1 = torch.randn(n_samples, 1)
    y1 = torch.randn(n_samples, 1)
    z1 = torch.randn(n_samples, 1)

    # Case 2: Three variables with pairwise independence but joint dependence
    # X, Y, Z are pairwise independent but X+Y+Z = 0
    x2 = torch.randn(n_samples, 1)
    y2 = torch.randn(n_samples, 1)
    z2 = -(x2 + y2)  # This creates joint dependence

    # Case 3: Four variables with complex dependencies
    # X and Y are independent, Z = X^2 + noise, W = Y^2 + noise
    x3 = torch.randn(n_samples, 1)
    y3 = torch.randn(n_samples, 1)
    z3 = x3**2 + 0.1 * torch.randn(n_samples, 1)
    w3 = y3**2 + 0.1 * torch.randn(n_samples, 1)

    # Case 4: Mixed types (continuous and discrete)
    x4 = torch.randn(n_samples, 1)
    y4 = torch.randint(0, 5, (n_samples, 1)).float()
    z4 = torch.randn(n_samples, 1)
    w4 = torch.randint(0, 3, (n_samples, 1)).float()

    # Case 5: Circular dependency
    # X and Y are dependent through a circular relationship
    x5 = torch.randn(n_samples, 1)
    y5 = torch.sin(x5) + 0.1 * torch.randn(n_samples, 1)
    z5 = torch.cos(y5) + 0.1 * torch.randn(n_samples, 1)

    # Case 6: High-dimensional data
    # Testing with higher dimensional variables
    x6 = torch.randn(n_samples, 3)
    y6 = torch.randn(n_samples, 3)
    z6 = x6 + y6 + 0.1 * torch.randn(n_samples, 3)

    # Case 7: Non-linear dependencies with multiple variables
    # Complex non-linear relationships between variables
    x7 = torch.randn(n_samples, 1)
    y7 = torch.randn(n_samples, 1)
    z7 = torch.sin(x7) * torch.cos(y7) + 0.1 * torch.randn(n_samples, 1)
    w7 = torch.exp(-(x7**2)) * torch.exp(-(y7**2)) + 0.1 * torch.randn(n_samples, 1)

    # Case 8: Categorical variables with different levels
    # Testing with categorical variables of different cardinalities
    x8 = torch.randint(0, 2, (n_samples, 1)).float()  # Binary
    y8 = torch.randint(0, 5, (n_samples, 1)).float()  # 5 levels
    z8 = torch.randint(0, 10, (n_samples, 1)).float()  # 10 levels
    w8 = torch.randint(0, 20, (n_samples, 1)).float()  # 20 levels

    # Save all data to temporary files
    data_files = {}
    for name, data in [
        ("x1", x1),
        ("y1", y1),
        ("z1", z1),
        ("x2", x2),
        ("y2", y2),
        ("z2", z2),
        ("x3", x3),
        ("y3", y3),
        ("z3", z3),
        ("w3", w3),
        ("x4", x4),
        ("y4", y4),
        ("z4", z4),
        ("w4", w4),
        ("x5", x5),
        ("y5", y5),
        ("z5", z5),
        ("x6", x6),
        ("y6", y6),
        ("z6", z6),
        ("x7", x7),
        ("y7", y7),
        ("z7", z7),
        ("w7", w7),
        ("x8", x8),
        ("y8", y8),
        ("z8", z8),
        ("w8", w8),
    ]:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            np.savetxt(f.name, data.numpy(), delimiter=",")
            data_files[name] = f.name

    return {
        "data": {
            "x1": x1,
            "y1": y1,
            "z1": z1,
            "x2": x2,
            "y2": y2,
            "z2": z2,
            "x3": x3,
            "y3": y3,
            "z3": z3,
            "w3": w3,
            "x4": x4,
            "y4": y4,
            "z4": z4,
            "w4": w4,
            "x5": x5,
            "y5": y5,
            "z5": z5,
            "x6": x6,
            "y6": y6,
            "z6": z6,
            "x7": x7,
            "y7": y7,
            "z7": z7,
            "w7": w7,
            "x8": x8,
            "y8": y8,
            "z8": z8,
            "w8": w8,
        },
        "files": data_files,
    }


def run_r_implementation(data_files):
    """Run R implementation and return results."""
    data_files = {k: v.replace("\\", "/") for k, v in data_files.items()}

    r_script = f"""
    # Force immediate output
    options(echo=FALSE)

    # Set library path to user's R library
    user_lib <- normalizePath(path.expand("~/R/library"), winslash = "/", mustWork = FALSE)
    .libPaths(c(user_lib, .libPaths()))

    # Load library with error handling
    if (!require("dHSIC")) {{
        stop("dHSIC package not found")
    }}
    
    # Read all data
    x1 <- read.csv("{data_files['x1']}", header=FALSE)
    y1 <- read.csv("{data_files['y1']}", header=FALSE)
    z1 <- read.csv("{data_files['z1']}", header=FALSE)
    
    x2 <- read.csv("{data_files['x2']}", header=FALSE)
    y2 <- read.csv("{data_files['y2']}", header=FALSE)
    z2 <- read.csv("{data_files['z2']}", header=FALSE)
    
    x3 <- read.csv("{data_files['x3']}", header=FALSE)
    y3 <- read.csv("{data_files['y3']}", header=FALSE)
    z3 <- read.csv("{data_files['z3']}", header=FALSE)
    w3 <- read.csv("{data_files['w3']}", header=FALSE)
    
    x4 <- read.csv("{data_files['x4']}", header=FALSE)
    y4 <- read.csv("{data_files['y4']}", header=FALSE)
    z4 <- read.csv("{data_files['z4']}", header=FALSE)
    w4 <- read.csv("{data_files['w4']}", header=FALSE)
    
    x5 <- read.csv("{data_files['x5']}", header=FALSE)
    y5 <- read.csv("{data_files['y5']}", header=FALSE)
    z5 <- read.csv("{data_files['z5']}", header=FALSE)
    
    x6 <- read.csv("{data_files['x6']}", header=FALSE)
    y6 <- read.csv("{data_files['y6']}", header=FALSE)
    z6 <- read.csv("{data_files['z6']}", header=FALSE)
    
    x7 <- read.csv("{data_files['x7']}", header=FALSE)
    y7 <- read.csv("{data_files['y7']}", header=FALSE)
    z7 <- read.csv("{data_files['z7']}", header=FALSE)
    w7 <- read.csv("{data_files['w7']}", header=FALSE)
    
    x8 <- read.csv("{data_files['x8']}", header=FALSE)
    y8 <- read.csv("{data_files['y8']}", header=FALSE)
    z8 <- read.csv("{data_files['z8']}", header=FALSE)
    w8 <- read.csv("{data_files['w8']}", header=FALSE)
    
    # Print data dimensions for debugging
    cat("Data dimensions:\\n")
    cat("x1:", dim(x1), "\\n")
    cat("y1:", dim(y1), "\\n")
    cat("z1:", dim(z1), "\\n")
    
    # Case 1: Three independent Gaussian variables
    cat("\\nRunning Case 1...\\n")
    result1 <- dhsic(list(x1, y1, z1), kernel=c("gaussian", "gaussian", "gaussian"))
    
    # Case 2: Three variables with pairwise independence but joint dependence
    cat("\\nRunning Case 2...\\n")
    result2 <- dhsic(list(x2, y2, z2), kernel=c("gaussian", "gaussian", "gaussian"))
    
    # Case 3: Four variables with complex dependencies
    cat("\\nRunning Case 3...\\n")
    result3 <- dhsic(list(x3, y3, z3, w3), kernel=c("gaussian", "gaussian", "gaussian", "gaussian"))
    
    # Case 4: Mixed types (continuous and discrete)
    cat("\\nRunning Case 4...\\n")
    result4 <- dhsic(list(x4, y4, z4, w4), kernel=c("gaussian", "discrete", "gaussian", "discrete"))
    
    # Case 5: Circular dependency
    cat("\\nRunning Case 5...\\n")
    result5 <- dhsic(list(x5, y5, z5), kernel=c("gaussian", "gaussian", "gaussian"))
    
    # Case 6: High-dimensional data
    cat("\\nRunning Case 6...\\n")
    result6 <- dhsic(list(x6, y6, z6), kernel=c("gaussian", "gaussian", "gaussian"))
    
    # Case 7: Non-linear dependencies with multiple variables
    cat("\\nRunning Case 7...\\n")
    result7 <- dhsic(list(x7, y7, z7, w7), kernel=c("gaussian", "gaussian", "gaussian", "gaussian"))
    
    # Case 8: Categorical variables with different levels
    cat("\\nRunning Case 8...\\n")
    result8 <- dhsic(list(x8, y8, z8, w8), kernel=c("discrete", "discrete", "discrete", "discrete"))
    
    # Print all results in a format that can be easily parsed
    cat("R_RESULTS_START\\n")
    cat("1", result1$dHSIC, paste(result1$bandwidth, collapse=","), "\\n", sep=",")
    cat("2", result2$dHSIC, paste(result2$bandwidth, collapse=","), "\\n", sep=",")
    cat("3", result3$dHSIC, paste(result3$bandwidth, collapse=","), "\\n", sep=",")
    cat("4", result4$dHSIC, paste(result4$bandwidth, collapse=","), "\\n", sep=",")
    cat("5", result5$dHSIC, paste(result5$bandwidth, collapse=","), "\\n", sep=",")
    cat("6", result6$dHSIC, paste(result6$bandwidth, collapse=","), "\\n", sep=",")
    cat("7", result7$dHSIC, paste(result7$bandwidth, collapse=","), "\\n", sep=",")
    cat("8", result8$dHSIC, paste(result8$bandwidth, collapse=","), "\\n", sep=",")
    cat("R_RESULTS_END\\n")
    
    # Force output flush
    flush.console()
    """

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False) as f:
        f.write(r_script.encode())
        r_script_file = f.name

    try:
        # Run R script with explicit error handling
        result = subprocess.run(
            ["Rscript", "--vanilla", r_script_file], capture_output=True, text=True
        )

        if result.returncode != 0:
            print("Error running R script:")
            print(result.stderr)
            return None

        print("R Output:")
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"Error executing R script: {str(e)}")
        return None
    finally:
        os.unlink(r_script_file)


def main():
    # Check if R is installed
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: R is not installed or not in PATH.")
        print("Please install R from https://www.r-project.org/")
        sys.exit(1)

    if not check_r_dhsic_installed():
        # Always try to install dHSIC first
        print("\nInstalling/Updating dHSIC package in R...")
        if not install_r_dhsic():
            print("\nFailed to install dHSIC package automatically.")
            print("Please install it manually by running the following in R:")
            print('install.packages("dHSIC", repos="https://cloud.r-project.org")')
            sys.exit(1)

    # Generate test data
    n_samples = 200
    test_data = generate_test_data(n_samples)
    data = test_data["data"]
    data_files = test_data["files"]

    try:
        # Run R implementation
        print("\nRunning R implementation...")
        r_output = run_r_implementation(data_files)
        if r_output is None:
            print("Failed to run R implementation. Exiting.")
            sys.exit(1)

        # Extract R results
        r_results = {}
        r_bandwidths = {}
        start_idx = r_output.find("R_RESULTS_START")
        end_idx = r_output.find("R_RESULTS_END")
        if start_idx != -1 and end_idx != -1:
            results_text = r_output[start_idx:end_idx].split("\n")[1:-1]
            for line in results_text:
                if line.strip():
                    try:
                        # Split on comma and remove any trailing commas
                        parts = line.strip().rstrip(",").split(",")
                        if len(parts) >= 3:  # case_num, dHSIC, bandwidths...
                            case_num = int(parts[0])
                            value = float(parts[1])
                            # Handle NA values in bandwidths
                            bandwidths = []
                            for bw in parts[2:]:
                                if bw.strip().upper() == "NA":
                                    bandwidths.append(None)
                                else:
                                    try:
                                        bandwidths.append(float(bw))
                                    except ValueError:
                                        bandwidths.append(None)
                            r_results[case_num] = value
                            r_bandwidths[case_num] = bandwidths
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line}")
                        print(f"Error details: {str(e)}")
                        continue

        # Run Python implementation
        print("\nRunning Python implementation...")

        # Case 1: Three independent Gaussian variables
        result1 = dhsic(
            [data["x1"], data["y1"], data["z1"]],
            kernel=["gaussian", "gaussian", "gaussian"],
        )

        # Case 2: Three variables with pairwise independence but joint dependence
        result2 = dhsic(
            [data["x2"], data["y2"], data["z2"]],
            kernel=["gaussian", "gaussian", "gaussian"],
        )

        # Case 3: Four variables with complex dependencies
        result3 = dhsic(
            [data["x3"], data["y3"], data["z3"], data["w3"]],
            kernel=["gaussian", "gaussian", "gaussian", "gaussian"],
        )

        # Case 4: Mixed types (continuous and discrete)
        result4 = dhsic(
            [data["x4"], data["y4"], data["z4"], data["w4"]],
            kernel=["gaussian", "discrete", "gaussian", "discrete"],
        )

        # Case 5: Circular dependency
        result5 = dhsic(
            [data["x5"], data["y5"], data["z5"]],
            kernel=["gaussian", "gaussian", "gaussian"],
        )

        # Case 6: High-dimensional data
        result6 = dhsic(
            [data["x6"], data["y6"], data["z6"]],
            kernel=["gaussian", "gaussian", "gaussian"],
        )

        # Case 7: Non-linear dependencies with multiple variables
        result7 = dhsic(
            [data["x7"], data["y7"], data["z7"], data["w7"]],
            kernel=["gaussian", "gaussian", "gaussian", "gaussian"],
        )

        # Case 8: Categorical variables with different levels
        result8 = dhsic(
            [data["x8"], data["y8"], data["z8"], data["w8"]],
            kernel=["discrete", "discrete", "discrete", "discrete"],
        )

        # Store Python results
        python_results = {
            1: result1["dHSIC"],
            2: result2["dHSIC"],
            3: result3["dHSIC"],
            4: result4["dHSIC"],
            5: result5["dHSIC"],
            6: result6["dHSIC"],
            7: result7["dHSIC"],
            8: result8["dHSIC"],
        }

        python_bandwidths = {
            1: result1["bandwidth"],
            2: result2["bandwidth"],
            3: result3["bandwidth"],
            4: result4["bandwidth"],
            5: result5["bandwidth"],
            6: result6["bandwidth"],
            7: result7["bandwidth"],
            8: result8["bandwidth"],
        }

        # Print dHSIC results side by side
        print("\nComparison of dHSIC Results:")
        print("\n{:<50} {:<15} {:<15}".format("Test Case", "R dHSIC", "Python dHSIC"))
        print("-" * 80)

        case_descriptions = {
            1: "Three Independent Gaussians",
            2: "Pairwise Independent but Jointly Dependent",
            3: "Four Variables with Complex Dependencies",
            4: "Mixed Types (Continuous and Discrete)",
            5: "Circular Dependency",
            6: "High-dimensional Data",
            7: "Non-linear Dependencies",
            8: "Categorical Variables with Different Levels",
        }

        for case_num in range(1, 9):
            print(
                "{:<50} {:<15.6f} {:<15.6f}".format(
                    f"Case {case_num}: {case_descriptions[case_num]}",
                    r_results[case_num],
                    python_results[case_num],
                )
            )

        # Print bandwidth results
        print("\nComparison of Bandwidth Values:")
        print(
            "\n{:<50} {:<30} {:<30}".format(
                "Test Case", "R Bandwidths", "Python Bandwidths"
            )
        )
        print("-" * 110)

        for case_num in range(1, 9):
            # Format R bandwidths, replacing None with 'NA'
            r_bw_str = ", ".join(
                "NA" if bw is None else f"{bw:.6f}" for bw in r_bandwidths[case_num]
            )
            # Format Python bandwidths
            if python_bandwidths[case_num] is not None:
                py_bw_str = ", ".join(
                    f"{bw:.6f}" if bw is not None else "NA"
                    for bw in python_bandwidths[case_num]
                )
            else:
                py_bw_str = "NA"
            print(
                "{:<50} {:<30} {:<30}".format(
                    f"Case {case_num}: {case_descriptions[case_num]}",
                    "[" + r_bw_str + "]",
                    "[" + py_bw_str + "]",
                )
            )

    finally:
        # Clean up temporary files
        for file_path in data_files.values():
            os.unlink(file_path)


if __name__ == "__main__":
    main()
