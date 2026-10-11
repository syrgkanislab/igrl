# Create a user library directory if it doesn't exist
user_lib <- normalizePath(path.expand("~/R/library"), winslash = "/", mustWork = FALSE)
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)

# Set the library path
.libPaths(c(user_lib, .libPaths()))

# Print R version and library paths
cat("R version:\n")
print(R.version.string)
cat("\nLibrary paths:\n")
print(.libPaths())

# Set up Rtools environment
cat("\nSetting up Rtools environment...\n")
rtools_path <- normalizePath("C:/rtools45", winslash = "/", mustWork = FALSE)
if (dir.exists(rtools_path)) {
    # Add Rtools to PATH
    path <- Sys.getenv("PATH")
    new_path <- paste0(
        normalizePath(file.path(rtools_path, "bin"), winslash = "/"), ";",
        normalizePath(file.path(rtools_path, "x86_64-w64-mingw32.static.posix/bin"), winslash = "/"), ";",
        normalizePath(file.path(rtools_path, "usr/bin"), winslash = "/"), ";",
        path
    )
    Sys.setenv(PATH = new_path)
    cat("Added Rtools to PATH\n")
} else {
    cat("Rtools not found at", rtools_path, "\n")
    cat("Please install Rtools from: https://cran.r-project.org/bin/windows/Rtools/\n")
    cat("After installing, you may need to restart your computer.\n")
    quit(status = 1)
}

# Verify make is available
make_path <- Sys.which("make")
if (make_path == "") {
    cat("Error: 'make' not found in PATH. Rtools installation may be incomplete.\n")
    cat("Please ensure Rtools is properly installed and added to PATH.\n")
    quit(status = 1)
} else {
    cat("Found make at:", make_path, "\n")
}

# Install required packages first
cat("\nInstalling required packages...\n")
required_packages <- c("Matrix", "Rcpp", "RcppArmadillo")
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat(sprintf("Installing %s...\n", pkg))
        tryCatch({
            install.packages(pkg, 
                           lib = user_lib,
                           repos = "https://cloud.r-project.org",
                           type = "source",
                           dependencies = TRUE)
        }, error = function(e) {
            cat(sprintf("Failed to install %s. Error: %s\n", pkg, conditionMessage(e)))
            quit(status = 1)
        })
    }
}

# Now install dHSIC
cat("\nInstalling dHSIC package...\n")
tryCatch({
    install.packages("dHSIC", 
                    lib = user_lib,
                    repos = "https://cloud.r-project.org",
                    type = "source",
                    dependencies = TRUE)
}, error = function(e) {
    cat("Installation failed. Error:", conditionMessage(e), "\n")
    quit(status = 1)
})

# Add the user library to the search path
.libPaths(c(user_lib, .libPaths()))

# Verify installation
cat("\nVerifying dHSIC installation:\n")
if (require("dHSIC")) {
    cat("dHSIC package successfully installed and loaded\n")
} else {
    cat("Failed to load dHSIC package\n")
    quit(status = 1)
} 