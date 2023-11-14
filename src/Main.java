import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.print("""
            This model will output a number near 1 if the first input is greater than the second, and a number near 0 otherwise.
            The model will be train on 100,000,000 random inputs, with 3 hidden neurons.
            
            Training neural network...\s""");

        // Create a new neural network with 2 inputs, 3 hidden neurons, and 1 output neuron
        NeuralNetwork nn = new NeuralNetwork(2, 3, 1);

        // Train the neural network with some data
        long startTime = System.nanoTime();
        for (int i = 0; i < 100000000; i++) {
            List<Double> inputs = Arrays.asList(Math.random(), Math.random());
            List<Double> expectedOutput = List.of(inputs.get(0) > inputs.get(1) ? 1.0 : 0.0);
            nn.train(inputs, expectedOutput, 0.1);
        }
        long endTime = System.nanoTime();
        System.out.println("Done in " + (endTime - startTime) / 1000000000 + " seconds.\n");

        System.out.println("Enter two numbers between 0 & 1 (it can be floating point numbers) to test the model.");

        // Get two inputs from the user
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter input 1: ");
        double input1 = Double.parseDouble(scanner.nextLine());
        System.out.print("Enter input 2: ");
        double input2 = Double.parseDouble(scanner.nextLine());

        // Test the neural network with some data
        List<Double> testInputs = Arrays.asList(input1, input2);
        List<Double> testOutput = nn.feedForward(testInputs);

        // Outputs 1 if the first input is greater than the second,
        // and 0 otherwise.
        System.out.println("\nTest inputs: " + testInputs);
        System.out.println("Test output: " + testOutput);
    }
}
