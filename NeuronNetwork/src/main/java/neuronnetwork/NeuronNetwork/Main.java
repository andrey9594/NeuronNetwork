package neuronnetwork.NeuronNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
	private final static String LEARN_PATH = "resources/learn_vectors.csv";
	private final static String CONTROL_PATH = "resources/control_vectors.csv";

	private final static String TEST_PATH = "resources/test_vectors.csv";
	private final static String OUTPUT_PATH = "resources/answer.csv";

	/**
	 * Count of vector's dimensional Will be counted in process
	 */
	private static int N = -1;
	private final static int M = 20;
	private final static int S = 5000;
	
	public final static String cuisine[] = {
			"irish", "mexican", "chinese", "filipino", "vietnamese", "spanish", "japanese", 
			"moroccan", "french", "greek", "indian", "jamaican", "british", "brazilian", "russian",
			"cajun_creole", "thai", "southern_us", "korean", "italian"
	};

	private static void testAndMakeOutput(MultilayerPerceptron perceptron) throws FileNotFoundException {
		Scanner sc = new Scanner(new File(TEST_PATH));
		try (PrintWriter pw = new PrintWriter(OUTPUT_PATH)) {
			pw.println("id,cuisine");
			while (sc.hasNextLine()) {
				String line = sc.nextLine();
				String s[] = line.split(","); // last num is ID!
				double x[] = new double[N];
				for (int i = 0; i < s.length - 1; i++) {
					int pos = Integer.parseInt(s[i]);
					if (pos < N)
						x[pos] = 1.0;
				}
				int currentExampleId = Integer.parseInt(s[s.length - 1]);
				double cuisines[] = perceptron.getResult(x); // or RBF!
				int cuisineMostLikelyId = -1;
				double mostLikelyCuisineValue = -1;
				for (int i = 0; i < cuisines.length; i++) {
					if (mostLikelyCuisineValue < cuisines[i]) {
						cuisineMostLikelyId = i;
						mostLikelyCuisineValue = cuisines[i];
					}
				}
				String cuisineName = cuisine[cuisineMostLikelyId];
				pw.println(currentExampleId + "," + cuisineName);
			}
			pw.flush();
		} finally {
			if (sc != null)
				sc.close();
		}
	}

	private static ArrayList<ArrayList<Integer>> readAll(String fileName) throws FileNotFoundException {
		ArrayList<ArrayList<Integer>> x = new ArrayList<>();
		Scanner sc = new Scanner(new File(fileName));
		try {
			while (sc.hasNextLine()) {
				String line = sc.nextLine();
				String stringValues[] = line.split(",");
				ArrayList<Integer> currentValues = new ArrayList<>();
				for (String s : stringValues)
					currentValues.add(Integer.parseInt(s));
				for (Integer value : currentValues)
					N = Math.max(value + 1, N);
				x.add(currentValues);
			}
		} finally {
			if (sc != null)
				sc.close();
		}
		return x;
	}

	public static void main(String[] args) throws FileNotFoundException {

		ArrayList<ArrayList<Integer>> positions = readAll(LEARN_PATH);
		ArrayList<ArrayList<Integer>> positionsControl = readAll(CONTROL_PATH);
		int examplesCount = positions.size();
		double x[][] = new double[examplesCount][N];
		double y[][] = new double[positions.size()][M]; 
		for (int i = 0; i < examplesCount; i++) {
			for (int j = 0; j < positions.get(i).size() - 1; j++)
				if (positions.get(i).get(j) < N)
					x[i][positions.get(i).get(j)] = 1;
			y[i][positions.get(i).get(positions.get(i).size() - 1)] = 1;
		}
		
		int examplesCountControl = positionsControl.size();
		double xc[][] = new double[examplesCountControl][N];
		double yc[][] = new double[positionsControl.size()][M]; 
		for (int i = 0; i < examplesCountControl; i++) {
			for (int j = 0; j < positionsControl.get(i).size() - 1; j++)
				if (positionsControl.get(i).get(j) < N)
					xc[i][positionsControl.get(i).get(j)] = 1;
			yc[i][positionsControl.get(i).get(positionsControl.get(i).size() - 1)] = 1;
		}
		
		MultilayerPerceptron perceptron = new MultilayerPerceptron(N, S, M);

		long timeMillis = 300000; // 600000 Это 10 мин
		double A = 0.1;
		double minE = 1e-3;
		boolean can = perceptron.fit(x, y, xc, yc, timeMillis, minE, A);
		System.out.println(can ? "*** Successfully has learned *** " : "*** Failure ***");
		System.out.println("Train error = " + perceptron.getAvgError(x, y) + ", control error = " + perceptron.getAvgError(xc, yc));
		//
		try {
			testAndMakeOutput(perceptron);
		} catch (FileNotFoundException e) {
			System.out.println("File is absent today");
			return;
		}

	}
}
