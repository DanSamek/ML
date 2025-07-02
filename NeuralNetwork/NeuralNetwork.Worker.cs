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
        
        private static readonly object _trainingLossLock = new();
        private readonly int _id;
        
        public Worker(NeuralNetwork network, int id)
        {
            _network = network;
            _id = id;
            
            _forwardContext = _network.Layers
                .Select(l => (new double[l.Size()], new double [l.Size()]))
                .ToList();
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

                var output = _forwardContext[^1].Activations.ToList(); // TODO arrays.
                var forwardResult = new ForwardResult(output, item.Expected);
                var loss = _network._lossFunction(forwardResult);
                
                Monitor.Enter(_trainingLossLock);
                    _network._trainingLoss += loss;
                Monitor.Exit(_trainingLossLock);
            }
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
                layerForwardContext.Activations[i] = activationFunction(layerForwardContext.Sums[i]);
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