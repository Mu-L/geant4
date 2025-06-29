//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// G4MultiUnion
//
// Class description:
//
//   An instance of "G4MultiUnion" constitutes a grouping of several solids.
//   The constituent solids are stored with their respective location in an
//   instance of "G4Node". An instance of "G4MultiUnion" is subsequently
//   composed of one or several nodes.

// 19.10.12 M.Gayer - Original implementation from USolids module
// 06.04.17 G.Cosmo - Adapted implementation in Geant4 for VecGeom migration
// --------------------------------------------------------------------
#ifndef G4MULTIUNION_HH
#define G4MULTIUNION_HH

#include <vector>

#include "G4VSolid.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4Point3D.hh"
#include "G4Vector3D.hh"
#include "G4SurfBits.hh"
#include "G4Voxelizer.hh"

class G4Polyhedron;

class G4MultiUnion : public G4VSolid
{
    friend class G4Voxelizer;

  public:

    G4MultiUnion() : G4VSolid("") {}
    G4MultiUnion(const G4String& name);
    ~G4MultiUnion() override;

    // Build the multiple union by adding nodes
    void AddNode(G4VSolid& solid, const G4Transform3D& trans);
    void AddNode(G4VSolid* solid, const G4Transform3D& trans);

    G4MultiUnion(const G4MultiUnion& rhs);
    G4MultiUnion& operator=(const G4MultiUnion& rhs);

    // Accessors
    inline const G4Transform3D& GetTransformation(G4int index) const;
    inline G4VSolid* GetSolid(G4int index) const;
    inline G4int GetNumberOfSolids()const;

    // Navigation methods
    EInside Inside(const G4ThreeVector& aPoint) const override;

    EInside InsideIterator(const G4ThreeVector& aPoint) const;

    // Safety methods
    G4double DistanceToIn(const G4ThreeVector& aPoint) const override;
    G4double DistanceToOut(const G4ThreeVector& aPoint) const override;
    inline void SetAccurateSafety(G4bool flag);

    // Exact distance methods
    G4double DistanceToIn(const G4ThreeVector& aPoint,
                          const G4ThreeVector& aDirection) const override;
    G4double DistanceToOut(const G4ThreeVector& aPoint,
                           const G4ThreeVector& aDirection,
                           const G4bool calcNorm = false,
                           G4bool* validNorm = nullptr,
                           G4ThreeVector* aNormalVector = nullptr) const override;

    G4double DistanceToInNoVoxels(const G4ThreeVector& aPoint,
                                  const G4ThreeVector& aDirection) const;
    G4double DistanceToOutVoxels(const G4ThreeVector& aPoint,
                                 const G4ThreeVector& aDirection,
                                 G4ThreeVector*       aNormalVector) const;
    G4double DistanceToOutVoxelsCore(const G4ThreeVector& aPoint,
                                     const G4ThreeVector& aDirection,
                                     G4ThreeVector*       aNormalVector,
                                     G4bool&           aConvex,
                                     std::vector<G4int>& candidates) const;
    G4double DistanceToOutNoVoxels(const G4ThreeVector& aPoint,
                                   const G4ThreeVector& aDirection,
                                   G4ThreeVector*       aNormalVector) const;

    G4ThreeVector SurfaceNormal(const G4ThreeVector& aPoint) const override;

    void Extent(EAxis aAxis, G4double& aMin, G4double& aMax) const;
    void BoundingLimits(G4ThreeVector& aMin, G4ThreeVector& aMax) const override;
    G4bool CalculateExtent(const EAxis pAxis,
                           const G4VoxelLimits& pVoxelLimit,
                           const G4AffineTransform& pTransform,
                           G4double& pMin, G4double& pMax) const override;
    G4double GetCubicVolume() override;
    G4double GetSurfaceArea() override;

    G4int GetNumOfConstituents() const override;
    G4bool IsFaceted() const override;

    G4VSolid* Clone() const override ;

    G4GeometryType GetEntityType() const override { return "G4MultiUnion"; }

    void Voxelize();
      // Finalize and prepare for use. User MUST call it once before
      // navigation use.

    EInside InsideNoVoxels(const G4ThreeVector& aPoint) const;
    inline G4Voxelizer& GetVoxels() const;

    std::ostream& StreamInfo(std::ostream& os) const override;

    G4ThreeVector GetPointOnSurface() const override;

    void DescribeYourselfTo ( G4VGraphicsScene& scene ) const override ;
    G4Polyhedron* CreatePolyhedron () const override ;
    G4Polyhedron* GetPolyhedron () const override;

    G4MultiUnion(__void__&);
      // Fake default constructor for usage restricted to direct object
      // persistency for clients requiring preallocation of memory for
      // persistifiable objects.

  private:

    EInside InsideWithExclusion(const G4ThreeVector& aPoint,
                                G4SurfBits* bits = nullptr) const;
    G4int SafetyFromOutsideNumberNode(const G4ThreeVector& aPoint,
                                      G4double& safety) const;
    G4double DistanceToInCandidates(const G4ThreeVector& aPoint,
                                    const G4ThreeVector& aDirection,
                                     std::vector<G4int>& candidates,
                                    G4SurfBits& bits) const;

    // Conversion utilities
    inline G4ThreeVector GetLocalPoint(const G4Transform3D& trans,
                                       const G4ThreeVector& gpoint) const;
    inline G4ThreeVector GetLocalVector(const G4Transform3D& trans,
                                       const G4ThreeVector& gvec) const;
    inline G4ThreeVector GetGlobalPoint(const G4Transform3D& trans,
                                       const G4ThreeVector& lpoint) const;
    inline G4ThreeVector GetGlobalVector(const G4Transform3D& trans,
                                       const G4ThreeVector& lvec) const;
    void TransformLimits(G4ThreeVector& min, G4ThreeVector& max,
                         const G4Transform3D& transformation) const;
  private:

    struct G4MultiUnionSurface
    {
      G4ThreeVector point;
      G4VSolid* solid;
    };

    std::vector<G4VSolid*> fSolids;
    std::vector<G4Transform3D> fTransformObjs;
    G4Voxelizer fVoxels;              // Vozelizer for the solid
    G4double fCubicVolume = 0.0;      // Cubic Volume
    G4double fSurfaceArea = 0.0;      // Surface Area
    G4double kRadTolerance;           // Cached radial tolerance
    mutable G4bool fAccurate = false; // Accurate safety (off by default)

    mutable G4bool fRebuildPolyhedron = false;
    mutable G4Polyhedron* fpPolyhedron = nullptr;
};

//______________________________________________________________________________
inline G4Voxelizer& G4MultiUnion::GetVoxels() const
{
  return (G4Voxelizer&)fVoxels;
}

//______________________________________________________________________________
inline const G4Transform3D& G4MultiUnion::GetTransformation(G4int index) const
{
  return fTransformObjs[index];
}

//______________________________________________________________________________
inline G4VSolid* G4MultiUnion::GetSolid(G4int index) const
{
  return fSolids[index];
}

//______________________________________________________________________________
inline G4int G4MultiUnion::GetNumberOfSolids() const
{
  return G4int(fSolids.size());
}

//______________________________________________________________________________
inline void G4MultiUnion::SetAccurateSafety(G4bool flag)
{
  fAccurate = flag;
}

//______________________________________________________________________________
inline
G4ThreeVector G4MultiUnion::GetLocalPoint(const G4Transform3D& trans,
                                          const G4ThreeVector& global) const
{
  // Returns local point coordinates converted from the global frame defined
  // by the transformation. This is defined by multiplying the inverse
  // transformation with the global vector.

  G4double px = global.x() - trans.dx();
  G4double py = global.y() - trans.dy();
  G4double pz = global.z() - trans.dz();
  G4double x = trans.xx()*px + trans.yx()*py + trans.zx()*pz;
  G4double y = trans.xy()*px + trans.yy()*py + trans.zy()*pz;
  G4double z = trans.xz()*px + trans.yz()*py + trans.zz()*pz;
  return { x, y, z };
}

//______________________________________________________________________________
inline
G4ThreeVector G4MultiUnion::GetLocalVector(const G4Transform3D& trans,
                                           const G4ThreeVector& global) const
{
  // Returns local point coordinates converted from the global frame defined
  // by the transformation. This is defined by multiplying the inverse
  // transformation with the global vector.

  G4double vx = global.x();
  G4double vy = global.y();
  G4double vz = global.z();
  G4double x = trans.xx()*vx + trans.yx()*vy + trans.zx()*vz;
  G4double y = trans.xy()*vx + trans.yy()*vy + trans.zy()*vz;
  G4double z = trans.xz()*vx + trans.yz()*vy + trans.zz()*vz;
  return { x, y, z };
}

//______________________________________________________________________________
inline
G4ThreeVector G4MultiUnion::GetGlobalPoint(const G4Transform3D& trans,
                                           const G4ThreeVector& local) const
{
  // Returns global point coordinates converted from the local frame defined
  // by the transformation. This is defined by multiplying this transformation
  // with the local vector.

  G4double px = local.x();
  G4double py = local.y();
  G4double pz = local.z();
  G4double x = trans.xx()*px + trans.xy()*py + trans.xz()*pz + trans.dx();
  G4double y = trans.yx()*px + trans.yy()*py + trans.yz()*pz + trans.dy();
  G4double z = trans.zx()*px + trans.zy()*py + trans.zz()*pz + trans.dz();
  return { x, y, z };
}

//______________________________________________________________________________
inline
G4ThreeVector G4MultiUnion::GetGlobalVector(const G4Transform3D& trans,
                                            const G4ThreeVector& local) const
{
  // Returns vector components converted from the local frame defined by the
  // transformation to the global one. This is defined by multiplying this
  // transformation with the local vector while ignoring the translation.

  G4double vx = local.x();
  G4double vy = local.y();
  G4double vz = local.z();
  G4double x = trans.xx()*vx + trans.xy()*vy + trans.xz()*vz;
  G4double y = trans.yx()*vx + trans.yy()*vy + trans.yz()*vz;
  G4double z = trans.zx()*vx + trans.zy()*vy + trans.zz()*vz;
  return { x, y, z };
}

#endif
