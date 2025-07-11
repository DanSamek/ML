using NUnit.Framework;
namespace ML.NeuralNetwork.Loader;

[TestFixture]
public class DataLoaderTests
{
    [Repeat(10000, true)]
    [TestCase(1)]
    [TestCase(2)]
    [TestCase(4)]
    [TestCase(8)]
    [TestCase(16)]
    public void ReadTest(int maxLoadedInMemory)
    {
        var data = new List<List<double>>
        {
            new() { 0.50, -0.506, 0.4045 },
            new() { 1, 1.9055, 1.1404 },
            new() { 2.1540, 1, 2.5405 },
            new() { 5.45406054, 3, 5.405600440 },
            new() { 6, 2.1206, 1 },
            new() {1.464647, 3.450697, -5 },
            new() {5.1, 506.525056484, 5.5 },
            new() { -6, -2.1206, 1 },
            new() {1.464647, -3.450697, -5 },
            new() {-5.1, 506.525056484, 5.5 },
            new() { 0.50, -0.506, -0.4045 },
            new() { 1, -1.9055, 1.1404 },
        };
        
        var fileName = NeuralNetworkTestBase.CreateFile(data);
        var dataLoader = new DataLoader(fileName, context => NeuralNetworkTestBase.Parse(context, 2), 2, 1, maxLoadedInMemory);

        while (true)
        {
            var item = dataLoader.GetNext();
            if (item is null)
                break;

            var inputAsList = item.Input.ToList();
            inputAsList.Add(item.Expected[0]);
            var found = data.Find(x => x.SequenceEqual(inputAsList));
            Assert.That(found, Is.Not.Null);
            
            NeuralNetworkArrayPool.Return(item);
            data.Remove(found!);
            
        }
        File.Delete(fileName);
        Assert.That(data.Count == 0);
    }
}