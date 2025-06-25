using NUnit.Framework;
using NUnit.Framework.Legacy;

namespace ML.LinearClassification;

[TestFixture]
public class LinearClassificationTest
{
    [Test]
    public void SimpleTest()
    {
        var result = LinearClassification.TryFindSolution([
            ([-1, -5], -1),
            ([-5, 1], -1),
            ([-4, 1], -1),
            ([-5, -3], -1),
            
            ([1, 1], 1),
            ([2, 1], 1),
            ([5, 3], 1),
            ([3, 5], 1),
            ([10, 1], 1),
        ], 50000, 5);

        ClassicAssert.AreEqual(result.Error, 0);
        Console.WriteLine($"x: {result.Vector[0]}, y: {result.Vector[1]}, c: {result.V0}");
        // x: 0.6241082061309502, y: 0.4838321797584155, c: 1.2533256561217359
        // 0.6 * x + 0.5 * y + 1.25 = 0
        // y = -1.2*x - 2.5
        
        result = LinearClassification.TryFindSolution([
            ([-1, -5], 1),
            ([-5, 1], 1),
            ([-4, 1], 1),
            ([-5, -3], 1),
            
            ([1, 1], -1),
            ([2, 1], -1),
            ([5, 3], -1),
            ([3, 5], -1),
            ([10, 1], -1),
        ], 50000, 5);
        
        ClassicAssert.AreEqual(result.Error, 0);
        Console.WriteLine($"x: {result.Vector[0]}, y: {result.Vector[1]}, c: {result.V0}");
    }
}