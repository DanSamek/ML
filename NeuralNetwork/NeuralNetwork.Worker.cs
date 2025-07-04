namespace ML.NeuralNetwork;
using static NeuralNetworkHelper;

public partial class NeuralNetwork
{
    /// <summary>
    /// Worker class for neural network forward + loss calculation.
    /// </summary>
    private class Worker{
        private readonly NeuralNetwork _network;
        private readonly List<(double[] Sums, double[] Activations)> _forwardContext;
        
        private readonly List<double[,]> _weightGradients = [];
        private readonly List<double[]> _biasGradients = [];
        private readonly List<double[]> _neuronGradiens = [];
        private readonly double[] _errors;
        
        private static readonly object _trainingLossLock = new();
        private readonly int _id; // Used for debugging mostly.
        
        public Worker(NeuralNetwork network, int id)
        {
            _network = network;
            _id = id;
            
            _forwardContext = _network.Layers
                .Select(l => (new double[l.Size()], new double [l.Size()]))
                .ToList();
            
            var inputLayerSize = network.InputLayer.Size();
            _weightGradients.Add(new double[inputLayerSize, _network.Layers[0].Size()]);
            for (var i = 0; i < _network.Layers.Count - 1; i++)
                _weightGradients.Add(new double[_network.Layers[i].Size(), _network.Layers[i + 1].Size()]);
            
            _biasGradients = _network.Layers
                .Select(l => new double[l.Size()])
                .ToList();
            
            _neuronGradiens = _network.Layers
                .Select(l => new double[l.Size()])
                .ToList();
            
            _errors = new double [_network.Layers[^1].Size()];
        }
        
        internal void Run(int totalWorkers)
        {
            while (true)
            {
                if (!_network._queue.TryDequeue(out var item))
                {
                    Interlocked.Increment(ref _network._waitingWorkers);
                    if (_network._waitingWorkers >= totalWorkers)
                    {
                        _network._emptyQueueEvent.Set();
                    }

                    _network._notEmptyQueueEvent.WaitOne();
                    Interlocked.Decrement(ref _network._waitingWorkers);
                    continue;
                }
                
                // null = stop
                if (item is null)
                    break;
                
                Forward(item.Input);
                
                for (var i = 0; i < _network.Layers[^1].Size(); i++)
                    _errors[i] = _network._lossFunction.Value(_forwardContext[^1].Activations[i], item.Expected[i]);
                
                Backpropagate(item);
            }
        }
        
        private void CalculateNeuronGradients(List<double> expected)
        {
            var outputLayer = _network.OutputLayer;
            var networkLossFunction = _network._lossFunction.Derivative;
            
            // dL/d ON = dL/dO * dO/dS 
            for (var i = 0; i < outputLayer.Size(); i++)
            {
                // dL/dO
                var lossDerivative = networkLossFunction(_forwardContext[^1].Activations[i], expected[i]);
                
                // dO/dS
                var activationFunctionDerivative = _network.OutputLayer.ActivationFunction.Derivative(_forwardContext[^1].Sums[i]);
                _neuronGradiens[^1][i] = lossDerivative * activationFunctionDerivative;
            }
            
            for (var i = _network.Layers.Count - 2; i >= 0; i--)
            {
                var nextLayerIdx = i + 1;
                var currentLayer = _network.Layers[i];
                
                // dL/Dn_i_j = Sum (k) ( _neuronGradient[i+1][k] * dNS / dCO * dCO/dCS  ) [NS = next sum, CO = current output, CS = current sum]
                //                   -> (_neuronGradient[i+1][k] * w [weight] * activationFunctionDerivative(CS)) 
                for (var j = 0; j < currentLayer.Size(); j++)
                {
                    var sum = 0.0;
                    var neuron = currentLayer.Neurons[j];
                    for (var k = 0; k < neuron.Weights.Count; k++)
                    {
                        sum += _neuronGradiens[nextLayerIdx][k] * neuron.Weights[k];
                    }
                    sum *= currentLayer.ActivationFunction.Derivative(_forwardContext[i].Sums[j]);
                    _neuronGradiens[i][j] = sum;
                }
                
            }
        }

        private void Backpropagate(TrainingItem item)
        {
            CalculateNeuronGradients(item.Expected);
            // NOTE: We do += for _weightGradients and _biasGradients.  
            
            // InputLayer -> FirstLayer
            for (var i = 0; i < _network.InputLayer.Size(); i++)
            {
                var totalWeights = _network.InputLayer.Features[0].Weights.Count;
                for (var w = 0; w < totalWeights; w++)
                    _weightGradients[0][i, w] += _neuronGradiens[0][w] * item.Input[i];
            }
            
            for (var i = 0; i < _network.Layers[0].Size(); i++)
                _biasGradients[0][i] += _neuronGradiens[0][i];
            
            // All hidden layers
            // for biases biasGradient[layerIdx][bidx] = _neuronGradiens[layerIdx][neuronIdx] * 1;
            // for weights weightGradient[layerIdx][neuronIdx][widx] = _neuronGradiens[layerIdx + 1][neuronIdx] * sum[layerIdx][neuronIdx]
            for (var i = 1; i < _network.Layers.Count - 1; i++)
            {
                var totalWeights = _network.Layers[i].Neurons[0].Weights.Count;
                for (var j = 0; j < _network.Layers[i].Size(); j++)
                {
                    for (var w = 0; w < totalWeights; w++)
                        _weightGradients[i][j, w] += _neuronGradiens[i + 1][w] * _forwardContext[i].Sums[j];
                    
                    _biasGradients[i][j] += _neuronGradiens[i][j];
                }
            }
            
            // TODO send to neural net - partial derivatives of entire net.
            // TODO net will do averages + weight & bias updates.
        }
        
        private void Forward(List<double> data)
        {
            ResetForwardContext();
        
            for (var i = 0; i < _network.InputLayer.Size(); i++)
            {
                var weights = _network.InputLayer.Features[i].Weights;
                for (var j = 0; j < _network.Layers[0].Size(); j++)
                    _forwardContext[0].Sums[j] += data[i] * weights[j];
            }

            Activate(0);
            for (var layer = 0; layer < _network.Layers.Count - 1; layer++)
            {
                var neurons = _network.Layers[layer].Neurons;

                for (var i = 0; i < neurons.Count; i++)
                {
                    var weights = _network.Layers[layer].Neurons[i].Weights;
                    for (var j = 0; j < _network.Layers[layer + 1].Size(); j++)
                    {
                        _forwardContext[layer + 1].Sums[j] += _forwardContext[layer].Activations[i] * weights[j];
                    }
                }
                Activate(layer + 1);
            }
        }

        private void Activate(int layerIdx)
        {
            var layer = _network.Layers[layerIdx];
            var activationFunction = layer.ActivationFunction;
            var layerSize = layer.Size();

            var layerForwardContext = _forwardContext[layerIdx];
            for (var i = 0; i < layerSize; i++)
            {
                layerForwardContext.Activations[i] = activationFunction.Value(layerForwardContext.Sums[i]);
            }
        }

        private void ResetForwardContext()  
        {
            for (var i = 0; i < _forwardContext.Count; i++)
            {
                var layer = _network.Layers[i];
                var layerSize = layer.Size();
                for (var j = 0; j < layerSize; j++) 
                {
                    _forwardContext[i].Sums[j] = layer.Neurons[j].Bias;
                    _forwardContext[i].Activations[j] = 0;
                }
            }
        }
    }
    
}