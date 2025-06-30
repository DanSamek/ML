namespace ML.NeuralNetwork;

/// <summary>
/// Used activation functions.
/// </summary>
public class ActivationFunctions
{
    public static double Sigmoid(double current) => 1 / (1 + Math.Exp(-current));

    public static double RELU(double current) => current <= 0 ? 0 : current;
    public static int CRELUMax { get; set; } = 300;
    public static double CRELU(double current) => double.Clamp(current, 0, CRELUMax);
}