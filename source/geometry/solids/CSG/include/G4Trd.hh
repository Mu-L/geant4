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
// G4Trd
//
// Class description:
//
//   A G4Trd is a trapezoid with the x and y dimensions varying along z
//   functions:
//
//   Member Data:
//
//     fDx1    Half-length along x at the surface positioned at -dz
//     fDx2    Half-length along x at the surface positioned at +dz
//     fDy1    Half-length along y at the surface positioned at -dz
//     fDy2    Half-length along y at the surface positioned at +dz
//     fDz     Half-length along z axis

// 12.01.95 P.Kent: Old prototype code converted to thick geometry
// 21.04.97 J.Apostolakis: Added Set Methods
// 19.11.99 V.Grichine: kUndefined was added to Eside enum
// --------------------------------------------------------------------
#ifndef G4TRD_HH
#define G4TRD_HH

#include "G4GeomTypes.hh"

#if defined(G4GEOM_USE_USOLIDS)
#define G4GEOM_USE_UTRD 1
#endif

#if defined(G4GEOM_USE_UTRD)
  #define G4UTrd G4Trd
  #include "G4UTrd.hh"
#else

#include "G4CSGSolid.hh"
#include "G4Polyhedron.hh"

class G4Trd : public G4CSGSolid
{
  public:

    G4Trd( const G4String& pName,
                 G4double pdx1, G4double pdx2,
                 G4double pdy1, G4double pdy2,
                 G4double pdz );
      //
      // Constructs a trapezoid with name, and half lengths

   ~G4Trd() override;
      //
      // Destructor

    // Accessors

    inline G4double GetXHalfLength1() const;
    inline G4double GetXHalfLength2() const;
    inline G4double GetYHalfLength1() const;
    inline G4double GetYHalfLength2() const;
    inline G4double GetZHalfLength()  const;

    // Modifiers

    inline void SetXHalfLength1(G4double val);
    inline void SetXHalfLength2(G4double val);
    inline void SetYHalfLength1(G4double val);
    inline void SetYHalfLength2(G4double val);
    inline void SetZHalfLength(G4double val);

    void SetAllParameters ( G4double pdx1, G4double pdx2,
                            G4double pdy1, G4double pdy2,
                            G4double pdz );

    // Methods of solid

    G4double GetCubicVolume() override;
    G4double GetSurfaceArea() override;

    void ComputeDimensions(       G4VPVParameterisation* p,
                            const G4int n,
                            const G4VPhysicalVolume* pRep ) override;

    void BoundingLimits(G4ThreeVector& pMin, G4ThreeVector& pMax) const override;

    G4bool CalculateExtent(const EAxis pAxis,
                           const G4VoxelLimits& pVoxelLimit,
                           const G4AffineTransform& pTransform,
                                 G4double& pMin, G4double& pMax) const override;

    EInside Inside( const G4ThreeVector& p ) const override;

    G4ThreeVector SurfaceNormal( const G4ThreeVector& p ) const override;

    G4double DistanceToIn( const G4ThreeVector& p,
                           const G4ThreeVector& v ) const override;

    G4double DistanceToIn( const G4ThreeVector& p ) const override;

    G4double DistanceToOut( const G4ThreeVector& p,
                            const G4ThreeVector& v,
                            const G4bool calcNorm = false,
                                  G4bool* validNorm = nullptr,
                                  G4ThreeVector* n = nullptr ) const override;

    G4double DistanceToOut( const G4ThreeVector& p ) const override;

    G4GeometryType GetEntityType() const override;

    G4ThreeVector GetPointOnSurface() const override;

    G4bool IsFaceted() const override;

    G4VSolid* Clone() const override;

    std::ostream& StreamInfo( std::ostream& os ) const override;

    // Visualisation functions

    void          DescribeYourselfTo (G4VGraphicsScene& scene) const override;
    G4Polyhedron* CreatePolyhedron   () const override;

    G4Trd(__void__&);
      // Fake default constructor for usage restricted to direct object
      // persistency for clients requiring preallocation of memory for
      // persistifiable objects.

    G4Trd(const G4Trd& rhs);
    G4Trd& operator=(const G4Trd& rhs);
      // Copy constructor and assignment operator

  private:

    void CheckParameters();
      // Check parameters

    void MakePlanes();
      // Set side planes

    G4ThreeVector ApproxSurfaceNormal( const G4ThreeVector& p ) const;
      // Algorithm for SurfaceNormal() following the original
      // specification for points not on the surface

  private:

    G4double halfCarTolerance;
    G4double fDx1,fDx2,fDy1,fDy2,fDz,fHx,fHy;
    struct { G4double a,b,c,d; } fPlanes[4];
};

#include "G4Trd.icc"

#endif

#endif
