namespace ML.LinearClassification;

/**
 * Classical equation for the plane.
 *      v^t * x + v_o = 0
 *      where v_o is in R
 *      v^t, x are in R^d.
 * 
 *  -> The goal is to find an equation, that "splits" d-dim space into 2 parts, that:
 *      -> all items that are marked +1 are on the one side (where normal vector is positive) and all with -1 are on the second side.
 *
 * Basically we want to minimalize an error:
 *   (sum E(sign(v^t * x + v_o))) / n
 *      Where E is equal to:
 *          -> 0 - the sign is same for item in the dataset.
 *              (sign(v^t * x + v_o) == item.expected)
 *          -> 1 - else.
 */
public static class LinearClassification
{
    public class Result
    {
        public List<double> Vector { get; set; } = [];
        public double V0 { get; set; }

        public double Error { get; set; } = double.MaxValue;
    }

    private static List<double> RandomVector(int dimension, int maximum)
    {
        var result = Enumerable.Range(0, dimension).Select(_ => Random.Shared.NextDouble() * Random.Shared.Next(-maximum, maximum)).ToList();
        return result;
    } 
    
    /// <summary>
    /// The easiest algorithm.
    ///     Generate k times random vector [v] and random value [v_0] and minimalize error.
    /// </summary>
    /// <param name="dataset">Dataset - (vector, (1 or -1)) </param>
    /// <param name="k">Hyperparameter - how many iterations should be done.</param>
    /// <param name="maximum"></param>
    /// <returns></returns>
    public static Result TryFindSolution(List<(List<double> item, int expected)> dataset, int k, int maximum = 1000)
    {
        var dimension =  dataset.First().item.Count;
        var result = new Result();
        
        for (var i = 0; i < k; i++)
        {
            var randomVector = RandomVector(dimension, maximum);
            var randomNumber = Random.Shared.NextDouble() * Random.Shared.Next(-maximum, maximum);
            var currentError = CalculateError(randomVector, randomNumber);

            if (currentError >= result.Error) continue;
            
            result.Vector =  randomVector;
            result.V0 = randomNumber;
            result.Error = currentError;
            
            if (result.Error == 0) break;
        }
        return result;

        double CalculateError(List<double> randomVector, double randomNumber)
        {
            var error = 0;
            foreach (var (item, expected) in dataset)
            {
                var matrixProduct = randomNumber;
                for (var i = 0; i < dimension; i++)
                {
                    matrixProduct += item[i] * randomVector[i];
                }
                
                var sign = Math.Sign(matrixProduct);
                if (sign == 0) sign = -1;

                if (expected == sign) continue;
                error += 1;
            }
            return error * 1.0 / dataset.Count;
        }
    }
}