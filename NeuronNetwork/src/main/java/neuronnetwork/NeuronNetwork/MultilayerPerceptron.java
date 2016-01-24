package neuronnetwork.NeuronNetwork;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.reflect.AnnotatedParameterizedType;
import java.util.Random;

/**
 * MLP
 */
public class MultilayerPerceptron {

	private Neuron network[][] = new Neuron[3][];
	private final int N; // input layer
	private int S; // hidden layer
	private final int M; // output layer

	/** epsilon for optimizations. @use method setEps */
	private double epsT = 0.1; // train
	private double epsC = 0.50; // control

	private PrintWriter debug;

	/**
	 * Constructor
	 * 
	 * @param N
	 * @param S
	 * @param M
	 */
	public MultilayerPerceptron(int N, int S, int M) {
		this.N = N;
		this.S = S;
		this.M = M;

		network = new Neuron[3][];
		network[2] = new Neuron[M];
		network[1] = new Neuron[S];
		network[0] = new Neuron[N];

		ActivationFunction F = new FunctionSigmoidLogistic();
		ActivationFunction Fid = new FunctionIdentity();
		NeuronKernelFunction f = new NeuronKernelFunctionSum();
		for (int i = 0; i < M; i++)
			network[2][i] = new Neuron(F, f, i, null, S);
		for (int i = 0; i < S; i++)
			network[1][i] = new Neuron(F, f, i, network[2], N);
		for (int i = 0; i < N; i++) {
			network[0][i] = new Neuron(Fid, f, i, network[1], 1);
			network[0][i].setWeight(0, 1);
		}

		try {
			debug = new PrintWriter("error.txt");
		} catch (FileNotFoundException e) {
			System.out.println("Can't create file for debug printing!");
			return;
		}
	}

	public void setEps(double epsT, double epsC) {
		this.epsT = epsT;
		this.epsC = epsC;
	}

	/**
	 * array[0] = epsT array[1] = epsC
	 * 
	 * @return array of epsilons
	 */
	public double[] getEps() {
		return new double[] { epsT, epsC };
	}

	private double getTheta(int epoch) {
		final double a = 1.5;
		final double b = 0.05;
		return Math.max(a * Math.exp(-(b * epoch)), 0.1);
	}

	/** backpropagation */
	private void backpropagationSample(double x[], double y[], double prevDeltaW[][][], int epoch) { 

		/** прямой проход */
		for (int i = 0; i < N; i++)
			network[0][i].setSignal(0, x[i]);
		for (int i = 0; i < N; i++)
			network[0][i].pushSignal();
		for (int i = 0; i < S; i++)
			network[1][i].pushSignal();

		/** обратный проход */
		double e[] = new double[M];
		double localGradOut[] = new double[M];
		for (int j = 0; j < M; j++) {
			e[j] = y[j] - network[2][j].getOut();
			localGradOut[j] = e[j] * network[2][j].getDerivateOut();
			for (int from = 0; from < S; from++) {
				double oldValue = network[2][j].getWeight(from);
				double alpha = 0.7;
				double theta = getTheta(epoch);
				double deltaW = theta * localGradOut[j] * network[1][from].getOut();
				double newValue = oldValue + deltaW + alpha * prevDeltaW[2][j][from];
				prevDeltaW[2][j][from] = deltaW;
				network[2][j].setWeight(from, newValue);
			}
		}

		for (int j = 0; j < S; j++) {
			double sum = 0;
			for (int k = 0; k < M; k++) {
				sum += localGradOut[k] * network[2][k].getWeight(j);
			}
			double sigma = network[1][j].getDerivateOut() * sum;
			for (int from = 0; from < N; from++) {
				double oldValue = network[1][j].getWeight(from);
				double theta = getTheta(epoch);
				double alpha = 0.7;
				double deltaW = theta * sigma * network[0][from].getOut();
				double newValue = oldValue + deltaW + alpha * prevDeltaW[1][j][from];
				prevDeltaW[1][j][from] = deltaW;
				network[1][j].setWeight(from, newValue);
			}
		}
	}

	/**
	 * @param count
	 *            number of generations
	 */
	private void geneticAlgorithm(int count, double x[][], double y[][]) {
		Random random = new Random();
		double error[] = new double[count];
		double anotherW[][][][] = new double[2][][][]; // layer,neuronNum,from,N
		anotherW[0] = new double[S][M][count];
		anotherW[1] = new double[M][S][count];
		for (int N = 0; N < count; N++) {
			for (int i = 0; i < M; i++)
				for (int j = 0; j < S; j++) {
					anotherW[1][i][j][N] = random.nextDouble() - 0.5;
					anotherW[0][j][i][N] = random.nextDouble() - 0.5;
					network[2][i].setWeight(j, anotherW[1][i][j][N]);
					network[1][j].setWeight(i, anotherW[0][j][i][N]);
				}
			for (int i = 0; i < x.length; i++) {
				double res[] = this.getResult(x[i]);
				for (int j = 0; j < res.length; j++)
					error[N] += Math.pow(res[j] - y[i][j], 2) / 2.;
			}
			error[N] /= (double) x.length;
		}
		double sumInvCoef = 0.;
		for (int N = 0; N < count; N++) {
			sumInvCoef += 1. / error[N];
		}
		double chanceToBeAlive[] = new double[count];
		for (int i = 0; i < count; i++) {
			chanceToBeAlive[i] = (1. / error[i]) / sumInvCoef;
		}
		int crossoverS = S / 3;
		int crossoverM = M / 3;
		double newAnotherW[][][][] = new double[2][][][];
		newAnotherW[0] = new double[S][M][count];
		newAnotherW[1] = new double[M][S][count];
		int bestNewIndex = -1;
		double bestError = Double.MAX_VALUE;
		for (int N = 0; N < count; N++) {
			double theNextParent1 = random.nextDouble();
			int index1 = 0;
			for (double sum = 0; sum < theNextParent1; sum += chanceToBeAlive[index1++])
				;
			if (index1 >= N)
				index1--;
			double theNextParent2 = random.nextDouble();
			int index2 = 0;
			for (double sum = 0; sum < theNextParent2; sum += chanceToBeAlive[index2++])
				;
			if (index2 >= N)
				index2--;
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < S; j++) {
					if (j < crossoverS)
						newAnotherW[1][i][j][N] = anotherW[1][i][j][index1];
					else
						newAnotherW[1][i][j][N] = anotherW[1][i][j][index2];
					this.network[2][i].setWeight(j, newAnotherW[1][i][j][N]);
				}
			}
			for (int i = 0; i < S; i++) {
				for (int j = 0; j < M; j++) {
					if (j < crossoverM)
						newAnotherW[0][i][j][N] = anotherW[0][i][j][index1];
					else
						newAnotherW[0][i][j][N] = anotherW[0][i][j][index2];
					this.network[1][i].setWeight(j, newAnotherW[0][i][j][N]);
				}
			}
			double newError = 0;
			for (int i = 0; i < x.length; i++) {
				double res[] = this.getResult(x[i]);
				for (int j = 0; j < res.length; j++)
					newError += Math.pow(res[j] - y[i][j], 2) / 2.;
			}
			newError /= (double) x.length;
			if (newError < bestError) {
				bestError = newError;
				bestNewIndex = N;
			}
		}
		for (int i = 0; i < M; i++)
			for (int j = 0; j < S; j++) {
				this.network[2][i].setWeight(j, newAnotherW[1][i][j][bestNewIndex]);
				this.network[1][j].setWeight(i, newAnotherW[0][j][i][bestNewIndex]);
			}

	}

	public double getAvgError(double x[][], double y[][]) {
		double error = 0;
		for (int i = 0; i < x.length; i++) {
			double res[] = this.getResult(x[i]);
			for (int j = 0; j < res.length; j++)
				error += Math.pow(res[j] - y[i][j], 2);
		}
		error /= (double) x.length;
		return error;
	}

	void printErrorData(double x[][], double y[][]) {
		debug.print(this.getAvgError(x, y) + " ");
	}

	/**
	 * 
	 * @param x
	 *            train vectors
	 * @param y
	 *            train answers
	 * @param xc
	 *            control vectors
	 * @param yc
	 *            control answers
	 * @param timeMillis
	 *            Max time for learning
	 * @param minE
	 *            <= means we have paralysis
	 * @param A
	 *            shake-up weights
	 */
	public boolean fit(double x[][], double y[][], double xc[][], double yc[][], long timeMillis, double minE,
			double A) {
		long startTime = System.currentTimeMillis();

		while (true) {
			System.out.println("Genetic begin...");
			geneticAlgorithm(100, x, y);
			System.out.println("Genetic end");
			int k = x.length;
			Random random = new Random();
			double prevDeltaW[][][] = new double[3][Math.max(S, Math.max(M, N))][Math.max(S, Math.max(M, N))];
			int epoch = 0;
			while (this.getAvgError(x, y) > epsT) {
				epoch++;
				// System.out.println(this.getAvgError(x, y));
				printErrorData(x, y);
				double errorWas = getAvgError(x, y);
				boolean used[] = new boolean[k];
				System.out.println("Epoch = " + epoch + ", train error = " + this.getAvgError(x, y));
				for (int i = 0; i < k; i++) {
					int nextNum = random.nextInt(k);
					while (used[nextNum]) {
						nextNum = random.nextInt(k);
					}
					used[nextNum] = true;
					backpropagationSample(x[nextNum], y[nextNum], prevDeltaW, epoch);
				}
				double errorNow = getAvgError(x, y);
				if (Math.abs(errorWas - errorNow) < minE) {
					// паралич, делаем встряску на A
					// System.out.println("Shaking-up");
					for (int i = 0; i < M; i++) {
						for (int j = 0; j < S; j++) {
							network[2][i].setWeight(j, network[2][i].getWeight(j) + A);
							network[1][j].setWeight(i, network[1][j].getWeight(i) + A);
						}
					}
				}
				if (System.currentTimeMillis() - startTime >= timeMillis) {
					return false;
				}
			}

			System.out.println(
					"Has learned: train error = " + this.getAvgError(x, y) + ", control = " + this.getAvgError(xc, yc));
			if (this.getAvgError(xc, yc) < epsC) {
				return true;
			} else {
				System.out.println("Deleting 1 neuron in hidden layer");
				// уменьшаем на 1 нейрон в скрытом слое
				S--;
				for (int i = 0; i < M; i++)
					network[2][i] = new Neuron(new FunctionSigmoidLogistic(), new NeuronKernelFunctionSum(), i, null,
							S);
			}
		}
	}

	/**
	 * Calculate result for the current weights
	 * 
	 * @param x
	 *            signals
	 * @return output results
	 */
	public double[] getResult(double x[]) {
		if (x.length != N)
			throw new IllegalArgumentException();

		for (int i = 0; i < N; i++)
			network[0][i].setSignal(0, x[i]);

		for (int i = 0; i < N; i++)
			network[0][i].pushSignal();
		for (int i = 0; i < S; i++)
			network[1][i].pushSignal();

		double result[] = new double[M];
		for (int i = 0; i < M; i++)
			result[i] = network[2][i].getOut();
		return result;
	}
}
