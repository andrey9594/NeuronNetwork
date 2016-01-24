package neuronnetwork.NeuronNetwork;

import java.util.Arrays;
import java.util.Random;

/**
 * @author andrey
 * Class implements 1 neuron work:
 * Type of model depends on type of Function
 * It's may be sigmoid function or RBF 	
 */ 
public class Neuron {  
	private ActivationFunction F;
	private NeuronKernelFunction f;
	
	private double x[];
	private double w[];
	
	private int numInLayer;
	private Neuron nextLayer[];
	
	/**
	 * Constructor
	 * @param F Type of Activation Function: for example sigmoid function
	 * @param f Type of Kernel Function: for example sum x*w function
	 * @param numInLayer Neuron's position in the current slayer 
	 * @param nextLayer NextLayer neurons
	 * @param prevCount Means count of neurons in the previous layer
	 */
	public Neuron(ActivationFunction F, NeuronKernelFunction f, int numInLayer, Neuron[] nextLayer, int prevCount) {
		this.F = F;
		this.f = f;
		this.numInLayer = numInLayer;
		this.nextLayer = nextLayer;
		w = new double[prevCount];
		Random random = new Random();
		for (int i = 0; i < prevCount; i++) {
			w[i] = random.nextDouble() - 0.5; // [-0.5,0.5]
		}
		x = new double[prevCount];
	}
	
	/**
	 * Set signal value (x) got from some neuron in previous layer
	 * @param from From which neuron got signal
	 * @param inX Signal value (x)
	 */
	public void setSignal(int from, double inX) {
		if (from >= x.length || from < 0)
			throw new IllegalArgumentException();
		x[from] = inX;
	}
	
	public double getWeight(int from) {
		return w[from];
	}
	
	public void setWeight(int from, double newValue) {
		if (from >= w.length || from < 0)
			throw new IllegalArgumentException();
		w[from] = newValue;
	}
	
	/**
	 * Push signal from that neuron to all neurons in next layer
	 */
	public void pushSignal() {
		double neuronKernelFunctionValue = f.get(x, -1, w); 
		for (Neuron nextNeuron : nextLayer) {
			nextNeuron.setSignal(numInLayer, F.get(neuronKernelFunctionValue));
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public double getOut() {
		return F.get(f.get(x, -1, w));
	}
	
	/**
	 * 
	 * @return
	 */
	public double getDerivateOut() {
		return F.getDerivative(f.get(x, -1, w));
	}
	
	void debugPrint() {
		for (int i = 0; i < w.length; i++) {
			System.out.println(i + "->" + numInLayer + " = " + w[i]);
		}
	}
}
