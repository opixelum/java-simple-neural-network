import java.util.*;

class Neuron {
    private double bias;
    private double output;
    private final List<Double> weights;

    public Neuron(int inputs) {
        this.weights = new ArrayList<>();
        for (int i = 0; i < inputs; i++) {
            this.weights.add(Math.random());
        }
        this.bias = Math.random();
    }

    public double activate(double input) {
        this.output = sigmoid(input * this.weights.get(0) + this.bias);
        return this.output;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public List<Double> getWeights() {
        return this.weights;
    }

    public double getBias() {
        return this.bias;
    }

    public double getOutput() {
        return this.output;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
