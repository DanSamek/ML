namespace ML.NeuralNetwork;
using static NeuralNetworkHelper;

public partial class NeuralNetwork
{
    /// <summary>
    /// Worker class for neural network forward + loss calculation.
    /// </summary>
    private class Worker{
        private record Context(List<Layer> Layers, InputLayer InputLayer);
        private readonly NeuralNetwork _network;
        private Context _context;
        
        private static readonly object _trainingLossLock = new();
        private readonly int _id;
        
        public Worker(NeuralNetwork network, int id)
        {
            _network = network;
            _context = new Context(_network._layers, _network._inputLayer);
            _id = id;
        }

        internal void UpdateNetwork()
        {
            _context = new Context(_network._layers, _network._inputLayer);   
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

                var output = _context.Layers[^1].Neurons.Select(x => x.Sum).ToList();
                var forwardResult = new ForwardResult(output, item.Expected);
                var loss = _network._lossFunction(forwardResult);
                
                Monitor.Enter(_trainingLossLock);
                    _network._trainingLoss += loss;
                Monitor.Exit(_trainingLossLock);
            }
        }
        
        private void Forward(List<double> data)
        {
            ResetNeuronSums();
        
            for (var i = 0; i < _context.InputLayer.Size(); i++)
            {
                var weights = _context.InputLayer.Features[i].Weights;
                for (var j = 0; j < _context.Layers[0].Size(); j++)
                    _context.Layers[0].Neurons[j].Sum += data[i] * weights[j];
            }
        
            _context.Layers[0].Activate();
            for (var layer = 0; layer < _context.Layers.Count - 1; layer++)
            {
                var neurons = _context.Layers[layer].Neurons;

                for (var i = 0; i < neurons.Count; i++)
                {
                    var weights = _context.Layers[layer].Neurons[i].Weights;
                    for (var j = 0; j < _context.Layers[layer + 1].Size(); j++)
                    {
                        _context.Layers[layer + 1].Neurons[j].Sum += neurons[i].Sum * weights[j];
                    }
                }
                _context.Layers[layer].Activate();
            }
            _context.Layers[^1].Activate();
        }
        
        private void ResetNeuronSums() => _context.Layers.ForEach(l => l.Neurons.ForEach(n => n.Sum = n.Bias));
    }
    
}