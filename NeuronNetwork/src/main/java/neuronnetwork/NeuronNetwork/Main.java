package neuronnetwork.NeuronNetwork;

/**
 * Hello world!
 *
 */
public class Main {
	public static void main(String[] args) {
		int N = 3;
		int M = 2;
		int S = 2;
		double x[] = new double[N];
		for (int i = 0; i < N; i++)
			x[i] = 1;
		
		MultilayerPerceptron perceptron = new MultilayerPerceptron(N, S, M);
		double out[] = perceptron.getResult(x);
		for (double o : out) 
			System.out.print(o + " ");
	//	FunctionSigmoidLogistic e = new FunctionSigmoidLogistic();
	//	System.out.println(e.get(1.8));
	}
}
