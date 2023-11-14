import java.util.ArrayList;
import java.util.List;

class NeuralNetwork {
    private final List<Layer> layers;

    public NeuralNetwork(int inputs, int hidden, int outputs) {
        this.layers = new ArrayList<>();
        this.layers.add(new Layer(hidden, inputs));
        this.layers.add(new Layer(outputs, hidden));
    }

    public List<Double> feedForward(List<Double> inputs) {
        for (Layer layer : this.layers) {
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }

    public void train(List<Double> inputs, List<Double> expectedOutput, double learningRate) {
        List<Double> output = feedForward(inputs);
        List<Double> outputError = calculateOutputError(output, expectedOutput);
        List<Double> hiddenError = calculateHiddenError(outputError);
        updateWeightsAndBiases(inputs, outputError, hiddenError, learningRate);
    }

    private List<Double> calculateOutputError(List<Double> output, List<Double> expectedOutput) {
        List<Double> outputError = new ArrayList<>();
        for (int i = 0; i < output.size(); i++) {
            double error = expectedOutput.get(i) - output.get(i);
            outputError.add(error);
        }
        return outputError;
    }

    private List<Double> calculateHiddenError(List<Double> outputError) {
        List<Double> hiddenError = new ArrayList<>();
        for (int i = 0; i < layers.get(0).getNeurons().size(); i++) {
            double error = 0;
            for (int j = 0; j < layers.get(1).getNeurons().size(); j++) {
                error += outputError.get(j) * layers.get(1).getNeurons().get(j).getWeights().get(i);
            }
            hiddenError.add(error);
        }
        return hiddenError;
    }

    private void updateWeightsAndBiases(List<Double> inputs, List<Double> outputError, List<Double> hiddenError, double learningRate) {
        for (int i = 0; i < layers.get(1).getNeurons().size(); i++) {
            Neuron neuron = layers.get(1).getNeurons().get(i);
            for (int j = 0; j < neuron.getWeights().size(); j++) {
                double delta = outputError.get(i) * neuron.sigmoidDerivative(neuron.getOutput());
                neuron.getWeights().set(j, neuron.getWeights().get(j) + learningRate * delta * layers.get(0).getNeurons().get(j).getOutput());
                neuron.setBias(neuron.getBias() + learningRate * delta);
            }
        }

        for (int i = 0; i < layers.get(0).getNeurons().size(); i++) {
            Neuron neuron = layers.get(0).getNeurons().get(i);
            for (int j = 0; j < neuron.getWeights().size(); j++) {
                double delta = hiddenError.get(i) * neuron.sigmoidDerivative(neuron.getOutput());
                neuron.getWeights().set(j, neuron.getWeights().get(j) + learningRate * delta * inputs.get(j));
                neuron.setBias(neuron.getBias() + learningRate * delta);
            }
        }
    }
}
