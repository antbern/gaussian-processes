use nalgebra as na;

pub struct GaussianProcess<K: GpKernel> {
    kernel: K,
    x: na::DVector<f64>,
    y: na::DVector<f64>,
    input_cov_matrix_inv: na::DMatrix<f64>,
    noise_sigma: f64,
}

pub trait GpKernel {
    fn compute(&self, x: f64, x2: f64) -> f64;

    fn compute_matrix(&self, x: &na::DVector<f64>, x2: &na::DVector<f64>) -> na::DMatrix<f64> {
        let mut matrix = na::DMatrix::zeros(x.len(), x2.len());
        for i in 0..x.len() {
            for j in 0..x2.len() {
                matrix[(i, j)] = self.compute(x[i], x2[j]);
            }
        }
        matrix //.transpose() // TOD, correct?
    }
}

/// Radial basis function kernel
pub struct RbfKernel {
    pub sigma: f64,
    pub length_scale: f64,
}

impl GpKernel for RbfKernel {
    fn compute(&self, x: f64, x2: f64) -> f64 {
        self.sigma * (-0.5 * (x - x2).powi(2) / self.length_scale.powi(2)).exp()
    }
}

/// Constant to add to make sure matrices are positive definite
const EPS: f64 = 1e-6;

impl<K: GpKernel> GaussianProcess<K> {
    pub fn new(
        x: &na::DVector<f64>,
        y: &na::DVector<f64>,
        kernel: K,
        noise_sigma: f64,
    ) -> GaussianProcess<K> {
        let k = kernel.compute_matrix(x, x)
            + na::DMatrix::identity(x.len(), x.len()) * (noise_sigma + EPS);
        let inverse = k.try_inverse().expect("should be invertible");

        GaussianProcess {
            kernel,
            x: x.clone(),
            y: y.clone(),
            input_cov_matrix_inv: inverse,
            noise_sigma,
        }
    }

    pub fn predict(&self, x: &na::DVector<f64>) -> (na::DVector<f64>, na::DVector<f64>) {
        // Compute the covariance matrix between the input and the training data (lower left)
        let k_star = self.kernel.compute_matrix(&self.x, x);
        // Compute the covariance matrix between the input and itself (lower right)
        let k_star_star = self.kernel.compute_matrix(x, x);

        // println!("K_star: {:?}", k_star);
        // println!("K_star_star: {:?}", k_star_star);
        // println!("Input cov matrix inv: {:?}", self.input_cov_matrix_inv);
        // println!("Y: {:?}", self.y);

        // TODO: figure out the issue with this, why do we need the additional transpose for k_star?
        let mean = &k_star.transpose() * &self.input_cov_matrix_inv * &self.y;
        // println!("Mean; {:?}", mean);

        let covariance = k_star_star - k_star.transpose() * &self.input_cov_matrix_inv * &k_star;
        let covariance =
            &covariance + na::DMatrix::identity(covariance.nrows(), covariance.ncols()) * EPS;

        let variance = covariance.diagonal();

        // println!("Variance: {:?}", variance);

        (mean, variance)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use na::{DMatrix, DVector};

    #[test]
    fn test_rbf_kernel_compute() {
        let kernel = RbfKernel {
            sigma: 1.0,
            length_scale: 1.0,
        };
        let result = kernel.compute(1.0, 2.0);
        assert!((result - 0.60653066).abs() < 1e-6);
    }

    #[test]
    fn test_rbf_kernel_compute_matrix() {
        let kernel = RbfKernel {
            sigma: 1.0,
            length_scale: 1.0,
        };
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let x2 = DVector::from_vec(vec![1.0, 2.0]);
        let result = kernel.compute_matrix(&x, &x2);
        let expected = na::DMatrix::from_vec(2, 2, vec![1.0, 0.60653066, 0.60653066, 1.0]);
        assert!((result - expected).abs().max() < 1e-6);
    }

    #[test]
    fn test_gaussian_process_new() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        let kernel = RbfKernel {
            sigma: 1.0,
            length_scale: 1.0,
        };
        let gp = GaussianProcess::new(&x, &y, kernel, 0.1);
        assert_eq!(gp.x, x);
        assert_eq!(gp.y, y);
    }

    #[test]
    fn test_gaussian_process_predict() {
        let x_train = DVector::from_vec(vec![1.0, 2.0]);
        let y_train = DVector::from_vec(vec![3.0, 4.0]);
        let kernel = RbfKernel {
            sigma: 1.0,
            length_scale: 1.0,
        };
        let gp = GaussianProcess::new(&x_train, &y_train, kernel, 0.0);

        let x_test = DVector::from_vec(vec![1.0]);
        let (mean, variance) = gp.predict(&x_test);

        assert!((mean[0] - 3.0).abs() < 1e-1);
        assert!(variance[0].abs() < 1e-1);
    }
}
