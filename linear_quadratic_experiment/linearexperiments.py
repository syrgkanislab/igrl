import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import gendata
import methods

def iter(i, m, type, returnfull=False):
    rs = i
    if type == 1:
        np.random.seed(rs)
        n = 10000
        m = m
        k = 4
        r = 4

        # A is full rank
        A = 0.1*np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, r)

        # t is vector
        t = np.random.randn(r, 1)

        g = lambda Z : Z @ A.T
        f = lambda D : D @ B.T
        theta = lambda D : D @ t
        U = lambda n, r: 10 * np.random.randn(n, r)
        V = lambda n, m: 10 * np.random.randn(n, m)
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 1 * np.random.randn(n, 1)

        data = gendata.gendata(g, f, theta, U, V, Q, n, m, k, r, rs)
    
    elif type == 2:
        np.random.seed(rs)
        n = 10000
        m = m
        k = 4
        r = 4

        # A is full rank
        A = 0.1*np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, r)

        # t is vector
        t = np.random.randn(r, 1)

        h=3
        E=np.random.randn(h, r)

        g = lambda Z : Z @ A.T
        f = lambda D : D @ B.T
        theta = lambda D : D @ t
        U = lambda n, r: 20 * np.random.uniform(-1, 1, (n, h)) @ E
        V = lambda n, m: 10 * np.random.randn(n, m)
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 1 * np.random.randn(n, 1)    

        data = gendata.gendata(g, f, theta, U, V, Q, n, m, k, r, rs)
    
    elif type == 3:
        np.random.seed(rs)
        n = 10000
        m = m
        k = 4
        r = 4

        # A is full rank
        A = 0.1*np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, r)

        # t is vector
        t = np.random.randn(r, 1)

        h=3
        E=np.random.randn(h, r)
        #print(E)

        h2=5
        F=np.random.randn(h2, m)
        #print(F)

        g = lambda Z : Z @ A.T
        f = lambda D : D @ B.T
        theta = lambda D : D @ t
        U = lambda n, r: 20 * np.random.uniform(-1, 1, (n, h)) @ E
        V = lambda n, m: 5 * np.random.randn(n, h2) @ F
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 1 * np.random.randn(n, 1)

        data = gendata.gendata(g, f, theta, U, V, Q, n, m, k, r, rs)

    else:
        print("Not supported")
        return
    
    
    # standarization
    XStandardScaler = StandardScaler().fit(data["X"])
    X = XStandardScaler.transform(data["X"])
    YStandardScaler = StandardScaler().fit(data["Y"])
    Y = YStandardScaler.transform(data["Y"])
    Z = StandardScaler().fit_transform(data["Z"])
    
    # mymodel
    mymodel = methods.LIRR(r)
    mymodel.fit(Z, X, Y)
    
    # pcamodel
    pcamodel = methods.PCAMethod(r)
    pcamodel.fit(Z, X, Y)
    
    # compare test data
    # mymodel
    D = mymodel.encode(XStandardScaler.transform(data["X_test"]))
    Dprime = D + 1*mymodel.gettheta().T/np.linalg.norm(mymodel.gettheta())
    Xprime = XStandardScaler.inverse_transform(mymodel.decode(Dprime))
    Yprime = theta(Xprime @ np.linalg.pinv(B.T))
    
    # pcamodel
    D = pcamodel.encode(XStandardScaler.transform(data["X_test"]))
    Dprime = D + 1*pcamodel.gettheta().T/np.linalg.norm(pcamodel.gettheta())
    Xprime = XStandardScaler.inverse_transform(pcamodel.decode(Dprime))
    Yprime_pca = theta(Xprime @ np.linalg.pinv(B.T))
    
    # compare train data
    # mymodel
    D = mymodel.encode(X)
    Dprime = D + 1*mymodel.gettheta().T/np.linalg.norm(mymodel.gettheta())
    Xprime = XStandardScaler.inverse_transform(mymodel.decode(Dprime))
    Yprime_train = theta(Xprime @ np.linalg.pinv(B.T))
    
    # pcamodel
    D = pcamodel.encode(X)
    Dprime = D + 1*pcamodel.gettheta().T/np.linalg.norm(pcamodel.gettheta())
    Xprime = XStandardScaler.inverse_transform(pcamodel.decode(Dprime))
    Yprime_pca_train = theta(Xprime @ np.linalg.pinv(B.T))
    
    assert Yprime_pca.shape != Yprime_pca_train.shape
    assert Yprime.shape != Yprime_train.shape
    
    if returnfull:
        return Yprime, Yprime_pca, Yprime_train, Yprime_pca_train, data["Y_test"], data["Y"]
    
    # Calculate metrics for LIRR test data
    lirr_test_positive_rate = np.mean((Yprime - data["Y_test"]) >= 0)
    lirr_test_mean_diff = np.mean(Yprime - data["Y_test"])
    lirr_test_median_diff = np.median(Yprime - data["Y_test"])
    
    # Calculate metrics for PCA test data
    pca_test_positive_rate = np.mean((Yprime_pca - data["Y_test"]) >= 0)
    pca_test_mean_diff = np.mean(Yprime_pca - data["Y_test"])
    pca_test_median_diff = np.median(Yprime_pca - data["Y_test"])
    
    # Calculate metrics for LIRR train data
    lirr_train_positive_rate = np.mean((Yprime_train - data["Y"]) >= 0)
    lirr_train_mean_diff = np.mean(Yprime_train - data["Y"])
    lirr_train_median_diff = np.median(Yprime_train - data["Y"])
    
    # Calculate metrics for PCA train data
    pca_train_positive_rate = np.mean((Yprime_pca_train - data["Y"]) >= 0)
    pca_train_mean_diff = np.mean(Yprime_pca_train - data["Y"])
    pca_train_median_diff = np.median(Yprime_pca_train - data["Y"])
    
    # Return a dictionary with all metrics for easier analysis
    return {
        "lirr_test": {
            "positive_rate": lirr_test_positive_rate,
            "mean_diff": lirr_test_mean_diff,
            "median_diff": lirr_test_median_diff
        },
        "pca_test": {
            "positive_rate": pca_test_positive_rate,
            "mean_diff": pca_test_mean_diff,
            "median_diff": pca_test_median_diff
        },
        "lirr_train": {
            "positive_rate": lirr_train_positive_rate,
            "mean_diff": lirr_train_mean_diff,
            "median_diff": lirr_train_median_diff
        },
        "pca_train": {
            "positive_rate": pca_train_positive_rate,
            "mean_diff": pca_train_mean_diff,
            "median_diff": pca_train_median_diff
        }
    }


def experiment(dgp, file):
     # Structure: all_results[method][metric][m][iteration] = value
    metrics = ["positive_rate", "mean_diff", "median_diff"]
    all_results = {
        "lirr_test": {metric: {} for metric in metrics},
        "pca_test": {metric: {} for metric in metrics},
        "lirr_train": {metric: {} for metric in metrics},
        "pca_train": {metric: {} for metric in metrics}
    }
    avg_results = {
        "lirr_test": {metric: [] for metric in metrics},
        "pca_test": {metric: [] for metric in metrics},
        "lirr_train": {metric: [] for metric in metrics},
        "pca_train": {metric: [] for metric in metrics}
    }

    ms = [50, 100, 500]  # Model dimensions
    title={"lirr_test":"LIRR", "lirr_train":"LIRR", \
       "pca_test":"PCA", "pca_train":"PCA", \
      }
    
    repeats = 100
    # Run experiments
    for m in ms:
        # Initialize storage for this dimension in all results
        for method in ["lirr_test", "pca_test", "lirr_train", "pca_train"]:
            for metric in metrics:
                all_results[method][metric][m] = []
        
        print(f"Running experiments for dimension m={m}")
        for i in tqdm(range(repeats)):
            # Get results for this iteration
            result = iter(i, m, type=dgp)
            
            # Store all metrics from the result
            for method in ["lirr_test", "pca_test", "lirr_train", "pca_train"]:
                for metric in metrics:
                    value = result[method][metric]
                    all_results[method][metric][m].append(value)
        
        # After all iterations for this m, calculate and store average metrics
        for method in ["lirr_test", "pca_test", "lirr_train", "pca_train"]:
            for metric in metrics:
                avg_value = np.mean(all_results[method][metric][m])
                avg_results[method][metric].append(avg_value)
    
    # plotting
    lst = list(avg_results.keys())
    plt.figure(figsize=(4 * len(ms), 4))
    for m_idx, m in enumerate(ms):
        file.write(f"\tavg improvement across {repeats} runs and its std for m={m}, dgp={dgp}\n")
        
        for k in lst:
            if "test" in k:
                y = np.array(all_results[k]["mean_diff"][m])
                file.write(f"\t{title[k]}\t{np.mean(y):.4f}({np.std(y):.4f})\n")
                
                plt.subplot(1, len(ms) + 1, m_idx + 1)
                plt.title(f"m={m}")
                plt.hist(y, alpha=0.5, label=title[k], density=True, 
                        bins=np.linspace(-15, 50, 60))
                
                # Add vertical line at 0
                if m == 20:
                    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                    plt.xlim(-15, 40)
                    plt.ylim(0, 0.3)
                else:
                    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                    plt.xlim(-15, 20)
                    plt.ylim(0, 0.4)
                
                plt.legend()
                plt.xlabel('Average Test Improvement')
                plt.ylabel('Density')

    # Add overall title and adjust layout
    plt.suptitle('Test Improvements Across Different m Values with DGP=' + str(dgp), y=1.02)
    plt.tight_layout()

    # Optional: Save the figure
    plt.savefig("plots/linear_avgimprove_dgp="+str(dgp)+".png", dpi=300, bbox_inches='tight')


def main():
    with open("linear_result.txt", "w") as file:
        file.write("Running experiment for DGP 1\n")
        experiment(1, file)
        file.write("Running experiment for DGP 2\n")
        experiment(2, file)
        file.write("Running experiment for DGP 3\n")
        experiment(3, file)

if __name__ == "__main__":
    main()