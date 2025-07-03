namespace ML.NeuralNetwork;

[Obsolete("Use ActivationFunctionBase child classes or custom ones")]
public class _ActivationFunctions
{
    // TODO
    public static int CreluMax { get; set; } = 256;
    public static double CRELU(double current) => double.Clamp(current, 0, CreluMax);
    
    // TODO
    public static int ScreluMax { get; set; } = 256;
    public static double SCRELU(double current) => Math.Pow(double.Clamp(current, 0, ScreluMax), 2);
}