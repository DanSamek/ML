namespace ML.NeuralNetwork.LossFunctions;

public class MAE : LossFunctionBase
{
    public double Value(double current, double expected) => Math.Abs(expected - current);
    
    public double Derivative(double current, double expected)
    {
        if (expected - current == 0)
            return 0;
        
        return expected - current < 0 ? -1 : 1;
    }
}