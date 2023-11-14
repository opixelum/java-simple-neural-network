import java.util.ArrayList;
import java.util.List;

class Layer {
    private final List<Neuron> neurons;

    public Layer(int numNeurons, int inputsPerNeuron) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            this.neurons.add(new Neuron(inputsPerNeuron));
        }
    }

    public List<Double> feedForward(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : this.neurons) {
            outputs.add(neuron.activate(inputs.get(0)));
        }
        return outputs;
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }
}
