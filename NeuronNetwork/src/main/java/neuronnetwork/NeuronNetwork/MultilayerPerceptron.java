package neuronnetwork.NeuronNetwork;

/**
 * @author andrey
 * 	
 */
public class MultilayerPerceptron {
	private Neuron network[][] = new Neuron[3][];
	
	private final int N; // enter layer
	private final int S; // hidden layer
	private final int M; // output layer
	
	/**
	 * Constructor
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
		NeuronKernelFunction f = new NeuronKernelFunctionSum();
		for (int i = 0; i < M; i++)
			network[2][i] = new Neuron(F, f, i, null, S);
		for (int i = 0; i < S; i++)
			network[1][i] = new Neuron(F, f, i, network[2], N);
		for (int i = 0; i < N; i++) 
			network[0][i] = new Neuron(F, f, i, network[1], 1);
	}		
	
	double [] getResult(double x[]) {
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
