// g++ -O3 -o nsdf nsdf.cpp -lopenvdb -I${HOME}/dev/include -L${HOME}/dev/lib && ./nsdf
// g++ -O3 -o nsdf nsdf.cpp -lopenvdb -ltbb -I${HOME}/dev/include -L${HOME}/dev/lib && ./nsdf ~/dev/data/vdb/vdbs/airplane/train/airplane_0001.vdb 1000

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

// computes a random number in a range from a to b
inline double RandomDouble(double a, double b) {
    const double random = ((double) rand()) / (double) RAND_MAX;
    return a + random * (b - a);
}

int main(int argc, char **argv)
{
    openvdb::initialize();// needed to read vdb files

    // parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " filename.vdb pointCount\n";
        return EXIT_FAILURE;
    }
    const int pointCount = atoi(argv[2]);
    const std::string fileName(argv[1]);

    // read vdb file
    openvdb::io::File file(fileName);
    file.open();
    openvdb::GridBase::Ptr baseGrid = file.getGrids()->at(0);
    file.close();
    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!grid) {
        std::cerr << "First  grid in " << argv[1] << " is not an SDF\n";
        return EXIT_FAILURE;
    }

    // Compute the world bounding box from the index bounding box
    const openvdb::CoordBBox indexBBox = grid->evalActiveVoxelBoundingBox();
    openvdb::math::BBox<openvdb::Vec3d> bbox;
    bbox.min() = openvdb::Vec3d(indexBBox.min()[0],   indexBBox.min()[1],   indexBBox.min()[2]);
    bbox.max() = openvdb::Vec3d(indexBBox.max()[0]+1, indexBBox.max()[1]+1, indexBBox.max()[2]+1);
    auto baseMap = grid->transform().baseMap();
    const auto worldBBox = bbox.applyMap(*baseMap);

    //std::cerr << indexBBox <<  "  " << worldBBox << std::endl;

    // sample signed distance values
    std::vector<openvdb::Vec3d> points(pointCount);
    std::vector<float>          values(pointCount); 
    openvdb::FloatGrid::Accessor acc = grid->getAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::Accessor, openvdb::tools::BoxSampler> sampler(acc, grid->transform());
    const float background = grid->background();
    for (int i=0, insidePoints = pointCount/2; i<insidePoints;) {
        openvdb::Vec3d xyz(RandomDouble(worldBBox.min().x(), worldBBox.max().x()),
                           RandomDouble(worldBBox.min().y(), worldBBox.max().y()),
                           RandomDouble(worldBBox.min().z(), worldBBox.max().z()));
        const float value = sampler.wsSample(xyz);
        if (value <= 0.0 && value > -background) {
            points[i] = xyz;
            values[i] = value;
            ++i;
        }
    }
    for (int i = pointCount/2; i<pointCount;) {
        openvdb::Vec3d xyz(RandomDouble(worldBBox.min().x(), worldBBox.max().x()),
                           RandomDouble(worldBBox.min().y(), worldBBox.max().y()),
                           RandomDouble(worldBBox.min().z(), worldBBox.max().z()));
        auto value = sampler.wsSample(xyz);
        if (value >= 0.0 && value < background) {
            points[i] = xyz;
            values[i] = value;
            ++i;
        }
    }

    // write results to a file
    std::ofstream os(fileName + ".txt");
    if (!os.is_open()) throw std::invalid_argument("Error opening file \""+fileName+".txt\"");
    os << pointCount << std::endl;
    for (int i=0; i<pointCount; ++i) {
        auto &xyz = points[i];
        os << xyz[0] << " " << xyz[1] << " " << xyz[2] << " " << values[i] << std::endl;
    }

    return EXIT_SUCCESS;
}